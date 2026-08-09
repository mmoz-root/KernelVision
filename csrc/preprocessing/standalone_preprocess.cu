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
        launch_naive(
            device_input,
            device_output,
            pixel_count,
            options.block_size);
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
            launch_naive(
                device_input,
                device_output,
                pixel_count,
                options.block_size);
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
        std::cout << "completed naive CUDA preprocessing: "
                  << options.height << 'x' << options.width << ' '
                  << options.dtype << " block=" << options.block_size << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_preprocess: " << error.what() << '\n';
        return 1;
    }
}
