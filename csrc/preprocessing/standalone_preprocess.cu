// Standalone raw-CUDA harness for KernelVision Milestone 5.
//
// This executable deliberately has no PyTorch/ATen/pybind dependency. Python
// provides an exact uint8 HWC fixture file, and this program writes a raw CHW
// output plus native CUDA-event samples.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Implementation {
    kNaive,
    kCoalesced,
    kWarpPacked,
};

Implementation parse_implementation(const std::string& name) {
    if (name == "naive") {
        return Implementation::kNaive;
    }
    if (name == "coalesced") {
        return Implementation::kCoalesced;
    }
    if (name == "warp_packed") {
        return Implementation::kWarpPacked;
    }
    throw std::invalid_argument(
        "implementation must be naive, coalesced, or warp_packed");
}

const char* implementation_name(Implementation implementation) {
    switch (implementation) {
        case Implementation::kNaive:
            return "naive";
        case Implementation::kCoalesced:
            return "coalesced";
        case Implementation::kWarpPacked:
            return "warp_packed";
    }
    throw std::invalid_argument("unsupported implementation");
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Options {
    int height = 0;
    int width = 0;
    int block_size = 256;
    int warmup_iterations = 1;
    int measured_iterations = 1;
    int launches_per_sample = 1;
    Implementation implementation = Implementation::kNaive;
    std::string dtype;
    std::string input_path;
    std::string output_path;
    std::string samples_path;
};

std::string require_value(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(
            std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--height") {
            options.height = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--width") {
            options.width = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--dtype") {
            options.dtype = require_value(argc, argv, index);
        } else if (argument == "--block-size") {
            options.block_size = std::stoi(require_value(argc, argv, index));
        } else if (argument == "--warmup") {
            options.warmup_iterations = std::stoi(
                require_value(argc, argv, index));
        } else if (argument == "--iterations") {
            options.measured_iterations = std::stoi(
                require_value(argc, argv, index));
        } else if (argument == "--launches-per-sample") {
            options.launches_per_sample = std::stoi(
                require_value(argc, argv, index));
        } else if (argument == "--implementation") {
            options.implementation = parse_implementation(
                require_value(argc, argv, index));
        } else if (argument == "--input") {
            options.input_path = require_value(argc, argv, index);
        } else if (argument == "--output") {
            options.output_path = require_value(argc, argv, index);
        } else if (argument == "--samples") {
            options.samples_path = require_value(argc, argv, index);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (options.height <= 0 || options.width <= 0) {
        throw std::invalid_argument("height and width must be positive");
    }
    if (options.dtype != "fp32" && options.dtype != "fp16") {
        throw std::invalid_argument("dtype must be fp32 or fp16");
    }
    if (options.block_size <= 0 || options.block_size > 1024) {
        throw std::invalid_argument("block-size must be in [1, 1024]");
    }
    if (options.warmup_iterations < 0 || options.measured_iterations <= 0) {
        throw std::invalid_argument(
            "warmup must be nonnegative and iterations must be positive");
    }
    if (options.launches_per_sample <= 0) {
        throw std::invalid_argument("launches-per-sample must be positive");
    }
    if (options.implementation == Implementation::kWarpPacked &&
        options.block_size % 32 != 0) {
        throw std::invalid_argument(
            "warp_packed requires block-size to be a multiple of 32");
    }
    if (options.input_path.empty() || options.output_path.empty() ||
        options.samples_path.empty()) {
        throw std::invalid_argument(
            "input, output, and samples paths are required");
    }
    return options;
}

std::vector<std::uint8_t> read_input(
    const std::string& path,
    std::size_t expected_bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("could not open input file: " + path);
    }
    const auto file_bytes = static_cast<std::size_t>(stream.tellg());
    if (file_bytes != expected_bytes) {
        throw std::runtime_error(
            "input byte count does not match height * width * 3");
    }
    stream.seekg(0);
    std::vector<std::uint8_t> input(expected_bytes);
    stream.read(
        reinterpret_cast<char*>(input.data()),
        static_cast<std::streamsize>(input.size()));
    if (!stream) {
        throw std::runtime_error("could not read complete input file");
    }
    return input;
}

template <typename Output>
__device__ Output convert_output(float value);

template <>
__device__ float convert_output<float>(float value) {
    return value;
}

template <>
__device__ __half convert_output<__half>(float value) {
    return __float2half_rn(value);
}

template <typename Output>
__global__ void coalesced_staged_bgr_hwc_to_rgb_chw(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count) {

    // Milestone 5 profile-guided optimization exercise:
    // 1. Declare a dynamic shared-memory byte tile.
    // 2. Compute this block's first global pixel and valid pixel count.
    // 3. Cooperatively copy the block's contiguous BGR bytes into the tile.
    // 4. Synchronize the entire block before any thread reads the tile.
    // 5. Only after the barrier, return threads outside pixel_count.
    // 6. Read this thread's B, G, R from shared memory.
    // 7. Normalize and store planar R, G, B exactly like the naive kernel.
    //
    // Important: every thread must reach __syncthreads(). A return before the
    // barrier can deadlock a partially full final block.

    extern __shared__ std::uint8_t tile[];

    const std::int64_t block_first_pixel = static_cast<std::int64_t>(blockIdx.x) * blockDim.x;
    const std::int64_t pixel_idx = block_first_pixel + threadIdx.x;
    const std::int64_t remaining_pixels = pixel_count - block_first_pixel;

    const std::int64_t valid_pixels = remaining_pixels < static_cast<std::int64_t>(blockDim.x)
        ? remaining_pixels
        : static_cast<std::int64_t>(blockDim.x);

    const std::int64_t tile_byte_count = valid_pixels * 3;
    const std::int64_t input_first_byte = block_first_pixel * 3;

    for(std::int64_t i = threadIdx.x; i < tile_byte_count; i += blockDim.x) {
        tile[i] = input[input_first_byte + i];
    }
    __syncthreads();

    if(pixel_idx >= pixel_count) return;

    const std::int64_t tile_pixel_idx = threadIdx.x;

    float b = tile[3*tile_pixel_idx];
    float g = tile[3*tile_pixel_idx + 1];
    float r = tile[3*tile_pixel_idx + 2];

    const float inverse_255 = 1.0f / 255.0f;

    b *= inverse_255;
    g *= inverse_255;
    r *= inverse_255;

    output[pixel_idx] = convert_output<Output>(r);
    output[pixel_idx+pixel_count] = convert_output<Output>(g);
    output[pixel_idx+pixel_count*2] = convert_output<Output>(b);
}

template <typename Output>
__global__ void warp_packed_bgr_hwc_to_rgb_chw(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count) {

    // Milestone 5 lower-overhead optimization exercise:
    // 1. Compute lane_id, warp_first_pixel, and this lane's pixel_idx.
    // 2. Use the naive scalar path for the final incomplete warp.
    // 3. For a full warp, lanes 0..23 each load one contiguous uint32 word.
    // 4. For each B/G/R byte needed by this lane, compute:
    //      source lane = byte offset / 4
    //      byte within word = byte offset % 4
    // 5. Fetch the source word with __shfl_sync and extract the byte.
    // 6. Normalize and store planar R/G/B using pixel_idx.
    //
    // No shared memory and no block-wide synchronization are used here.
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_in_block = threadIdx.x / warpSize;

    const std::int64_t block_first_pixel =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x;

    const std::int64_t warp_first_pixel =
        block_first_pixel +
        static_cast<std::int64_t>(warp_in_block) * warpSize;

    const std::int64_t pixel_idx = warp_first_pixel + lane_id;

    const bool full_warp = warp_first_pixel + (warpSize - 1) < pixel_count;

    if (!full_warp) {
        if (pixel_idx >= pixel_count) return;

        float b = input[3*pixel_idx];
        float g = input[3*pixel_idx + 1];
        float r = input[3*pixel_idx + 2];

        const float inverse_255 = 1.0f / 255.0f;

        b *= inverse_255;
        g *= inverse_255;
        r *= inverse_255;

        output[pixel_idx] = convert_output<Output>(r);
        output[pixel_idx+pixel_count] = convert_output<Output>(g);
        output[pixel_idx+pixel_count*2] = convert_output<Output>(b);
        return;
    }

    const std::int64_t warp_first_byte = warp_first_pixel * 3;

    std::uint32_t packed_word = 0;

    if (lane_id < 24) {
        const auto* packed_input =
            reinterpret_cast<const std::uint32_t*>(
                input + warp_first_byte);

        packed_word = packed_input[lane_id];
    }


    const unsigned int pixel_byte_offset = lane_id * 3;

    const unsigned int b_byte_offset = pixel_byte_offset;
    const unsigned int b_source_lane = b_byte_offset / 4;
    const unsigned int b_position = b_byte_offset % 4;
    const std::uint32_t b_word = __shfl_sync(
        0xFFFFFFFFu,
        packed_word,
        b_source_lane
    );
    float b = static_cast<float>(
        (b_word >> (b_position * 8u)) & 0xFFu
    );

    const unsigned int g_byte_offset = pixel_byte_offset + 1;
    const unsigned int g_source_lane = g_byte_offset / 4;
    const unsigned int g_position = g_byte_offset % 4;
    const std::uint32_t g_word = __shfl_sync(
        0xFFFFFFFFu,
        packed_word,
        g_source_lane
    );
    float g = static_cast<float>(
        (g_word >> (g_position * 8u)) & 0xFFu
    );

    const unsigned int r_byte_offset = pixel_byte_offset + 2;
    const unsigned int r_source_lane = r_byte_offset / 4;
    const unsigned int r_position = r_byte_offset % 4;
    const std::uint32_t r_word = __shfl_sync(
        0xFFFFFFFFu,
        packed_word,
        r_source_lane
    );
    float r = static_cast<float>(
        (r_word >> (r_position * 8u)) & 0xFFu
    );

    const float inverse_255 = 1.0f / 255.0f;

    b *= inverse_255;
    g *= inverse_255;
    r *= inverse_255;

    output[pixel_idx] =
        convert_output<Output>(r);
    output[pixel_idx + pixel_count] =
        convert_output<Output>(g);
    output[pixel_idx + pixel_count * 2] =
        convert_output<Output>(b);
}

template <typename Output>
__global__ void naive_bgr_hwc_to_rgb_chw(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count) {

    // Milestone 5 exercise boundary:
    // 1. Compute one global pixel index from blockIdx/threadIdx.
    // 2. Return when the pixel index is outside pixel_count.
    // 3. Load B, G, R from interleaved input offsets 3*i + {0,1,2}.
    // 4. Normalize each value by 255.0f.
    // 5. Store R, G, B at planar offsets i, P+i, and 2*P+i.
    // Use convert_output<Output>(...) so this template supports FP32 and FP16.

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

    output[pixel_idx] = convert_output<Output>(r);
    output[pixel_idx+pixel_count] = convert_output<Output>(g);
    output[pixel_idx+pixel_count*2] = convert_output<Output>(b);

}

template <typename Output>
void launch_naive(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count,
    int block_size) {
    const auto grid_size = static_cast<unsigned int>(
        (pixel_count + block_size - 1) / block_size);
    naive_bgr_hwc_to_rgb_chw<Output><<<grid_size, block_size>>>(
        input,
        output,
        pixel_count);
}

template <typename Output>
void launch_coalesced(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count,
    int block_size) {
    const auto grid_size = static_cast<unsigned int>(
        (pixel_count + block_size - 1) / block_size);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(block_size) * 3;
    coalesced_staged_bgr_hwc_to_rgb_chw<Output>
        <<<grid_size, block_size, shared_bytes>>>(
            input,
            output,
            pixel_count);
}

template <typename Output>
void launch_warp_packed(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count,
    int block_size) {
    const auto grid_size = static_cast<unsigned int>(
        (pixel_count + block_size - 1) / block_size);
    warp_packed_bgr_hwc_to_rgb_chw<Output><<<grid_size, block_size>>>(
        input,
        output,
        pixel_count);
}

template <typename Output>
void launch_preprocess(
    const std::uint8_t* input,
    Output* output,
    std::int64_t pixel_count,
    int block_size,
    Implementation implementation) {
    switch (implementation) {
        case Implementation::kNaive:
            launch_naive(input, output, pixel_count, block_size);
            return;
        case Implementation::kCoalesced:
            launch_coalesced(input, output, pixel_count, block_size);
            return;
        case Implementation::kWarpPacked:
            launch_warp_packed(input, output, pixel_count, block_size);
            return;
    }
    throw std::invalid_argument("unsupported implementation");
}

template <typename Output>
void run(const Options& options) {
    const std::int64_t pixel_count =
        static_cast<std::int64_t>(options.height) * options.width;
    const std::size_t input_bytes =
        static_cast<std::size_t>(pixel_count) * 3;
    const std::size_t output_count =
        static_cast<std::size_t>(pixel_count) * 3;
    const std::size_t output_bytes = output_count * sizeof(Output);

    const auto host_input = read_input(options.input_path, input_bytes);
    std::vector<Output> host_output(output_count);
    std::uint8_t* device_input = nullptr;
    Output* device_output = nullptr;
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes),
        "cudaMalloc input");
    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes),
        "cudaMalloc output");
    check_cuda(
        cudaMemcpy(
            device_input,
            host_input.data(),
            input_bytes,
            cudaMemcpyHostToDevice),
        "copy input to device");

    for (int iteration = 0; iteration < options.warmup_iterations; ++iteration) {
        launch_preprocess(
            device_input,
            device_output,
            pixel_count,
            options.block_size,
            options.implementation);
        check_cuda(cudaGetLastError(), "warmup kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    std::vector<cudaEvent_t> starts(options.measured_iterations);
    std::vector<cudaEvent_t> stops(options.measured_iterations);
    for (int iteration = 0; iteration < options.measured_iterations; ++iteration) {
        check_cuda(cudaEventCreate(&starts[iteration]), "create start event");
        check_cuda(cudaEventCreate(&stops[iteration]), "create stop event");
        check_cuda(cudaEventRecord(starts[iteration]), "record start event");
        for (int launch = 0; launch < options.launches_per_sample; ++launch) {
            launch_preprocess(
                device_input,
                device_output,
                pixel_count,
                options.block_size,
                options.implementation);
        }
        check_cuda(cudaEventRecord(stops[iteration]), "record stop event");
        check_cuda(cudaGetLastError(), "measured kernel launches");
    }
    check_cuda(
        cudaEventSynchronize(stops.back()),
        "measured kernel synchronization");

    std::ofstream sample_stream(options.samples_path);
    if (!sample_stream) {
        throw std::runtime_error(
            "could not open samples file: " + options.samples_path);
    }
    sample_stream << "iteration,latency_ms\n";
    sample_stream << std::setprecision(9);
    for (int iteration = 0; iteration < options.measured_iterations; ++iteration) {
        float elapsed_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(
                &elapsed_ms,
                starts[iteration],
                stops[iteration]),
            "calculate elapsed time");
        const float latency_per_launch_ms =
            elapsed_ms / options.launches_per_sample;
        sample_stream << iteration << ',' << latency_per_launch_ms << '\n';
        check_cuda(cudaEventDestroy(starts[iteration]), "destroy start event");
        check_cuda(cudaEventDestroy(stops[iteration]), "destroy stop event");
    }

    check_cuda(
        cudaMemcpy(
            host_output.data(),
            device_output,
            output_bytes,
            cudaMemcpyDeviceToHost),
        "copy output to host");
    std::ofstream output_stream(options.output_path, std::ios::binary);
    if (!output_stream) {
        throw std::runtime_error(
            "could not open output file: " + options.output_path);
    }
    output_stream.write(
        reinterpret_cast<const char*>(host_output.data()),
        static_cast<std::streamsize>(output_bytes));
    if (!output_stream) {
        throw std::runtime_error("could not write complete output file");
    }

    check_cuda(cudaFree(device_input), "free input");
    check_cuda(cudaFree(device_output), "free output");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.dtype == "fp32") {
            run<float>(options);
        } else {
            run<__half>(options);
        }
        std::cout << "completed "
                  << implementation_name(options.implementation)
                  << " CUDA preprocessing: "
                  << options.height << 'x' << options.width << ' '
                  << options.dtype << " block=" << options.block_size << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_preprocess: " << error.what() << '\n';
        return 1;
    }
}
