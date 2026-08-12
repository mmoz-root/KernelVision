// PyTorch-aware CUDA launcher and fused preprocessing kernel.

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>

bool cuda_translation_unit_loaded() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

template<typename Output>
__global__
void bgr_hwc_to_rgb_chw_k(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count
) {
    std::int64_t pixel_idx = static_cast<std::int64_t>(
        blockIdx.x) * blockDim.x + threadIdx.x;

    if(pixel_idx >= pixel_count) return;

    float b = input[3*pixel_idx];
    float g = input[3*pixel_idx + 1];
    float r = input[3*pixel_idx + 2];

    float norm = 1 / 255.0f;

    b *= norm;
    g *= norm;
    r *= norm;

    output[pixel_idx] = static_cast<Output>(r);
    output[pixel_idx+pixel_count] = static_cast<Output>(g);
    output[pixel_idx+pixel_count*2] = static_cast<Output>(b);

}


torch::Tensor preprocess_cuda(
    torch::Tensor image,
    bool output_fp16) {

    TORCH_CHECK(
        image.is_cuda(),
        "image must be on cuda device"
    );

    TORCH_CHECK(
        image.dim() == 3,
        "image must have hwc shape / 3 dimensions"
    );

    TORCH_CHECK(
        image.size(2) == 3,
        "image must have 3 channels"
    );

    TORCH_CHECK(
        image.scalar_type() == torch::kUInt8,
        "image must have torch.uint8 dtype"
    );

    TORCH_CHECK(
        image.is_contiguous(),
        "image must be contiguous"
    );

    TORCH_CHECK(
        image.size(0) > 0 && image.size(1) > 0,
        "image height and width must be positive"
    );

    const c10::cuda::CUDAGuard device_guard(image.device());

    const std::int64_t height = image.size(0);
    const std::int64_t width = image.size(1);

    const std::int64_t pixel_count = height * width;
    constexpr int threads = 256;

    const unsigned int blocks = static_cast<unsigned int>(
        (pixel_count + threads - 1) / threads
    );


    const torch::ScalarType output_dtype = output_fp16 ? torch::kFloat16 : torch::kFloat32;

    torch::Tensor output = torch::empty(
        {3, height, width},
        image.options().dtype(output_dtype)
    );

    const std::uint8_t* input_pointer = image.data_ptr<std::uint8_t>();

    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
        image.get_device()
    ).stream();

    if (output_fp16) {
        bgr_hwc_to_rgb_chw_k<at::Half>
            <<<blocks, threads, 0, stream>>>(
                input_pointer,
                output.data_ptr<at::Half>(),
                pixel_count
            );
    } else {
        bgr_hwc_to_rgb_chw_k<float>
            <<<blocks, threads, 0, stream>>>(
                input_pointer,
                output.data_ptr<float>(),
                pixel_count
            );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;

}
