// PyTorch binding for KernelVision fused CUDA preprocessing.

#include <torch/extension.h>

bool cuda_translation_unit_loaded();

torch::Tensor preprocess_cuda(
    torch::Tensor image,
    bool output_fp16
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "preprocess",
        &preprocess_cuda,
        "Run fused CUDA preprocessing",
        pybind11::arg("image"),
        pybind11::arg("output_fp16")
    );
    module.def(
        "_cuda_translation_unit_loaded",
        &cuda_translation_unit_loaded,
        "Return whether the CUDA translation unit can see a CUDA device"
    );
}
