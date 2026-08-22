"""Reusable raw TensorRT engine backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TensorRTBackend:
    """Load a TensorRT engine once and execute it repeatedly."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        path = Path(engine_path)
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

        try:
            import tensorrt as trt
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "TensorRT and PyTorch are required for TensorRT inference."
            ) from error

        self._trt = trt
        self._torch = torch
        self._device = torch.device(device)

        if self._device.type != "cuda":
            raise ValueError("TensorRT requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires an available CUDA device")

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(
            path.read_bytes()
        )

        if self._engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("failed to create TensorRT execution context")

        io_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        ]

        input_names = [
            name
            for name in io_names
            if self._engine.get_tensor_mode(name)
            == trt.TensorIOMode.INPUT
        ]
        output_names = [
            name
            for name in io_names
            if self._engine.get_tensor_mode(name)
            == trt.TensorIOMode.OUTPUT
        ]

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "expected exactly one TensorRT input and output"
            )

        self._input_name = input_names[0]
        self._output_name = output_names[0]

        self._input_shape = tuple(
            self._engine.get_tensor_shape(self._input_name)
        )
        self._output_shape = tuple(
            self._engine.get_tensor_shape(self._output_name)
        )
        if any(dimension < 0 for dimension in self._input_shape):
            raise ValueError("dynamic TensorRT input shapes are not supported")
        if any(dimension < 0 for dimension in self._output_shape):
            raise ValueError("dynamic TensorRT output shapes are not supported")

        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }

        engine_input_dtype = self._engine.get_tensor_dtype(
            self._input_name
        )
        engine_output_dtype = self._engine.get_tensor_dtype(
            self._output_name
        )

        if engine_input_dtype not in dtype_map:
            raise ValueError(
                f"unsupported input dtype: {engine_input_dtype}"
            )
        if engine_output_dtype not in dtype_map:
            raise ValueError(
                f"unsupported output dtype: {engine_output_dtype}"
            )

        self._input_dtype = dtype_map[engine_input_dtype]
        self._output_dtype = dtype_map[engine_output_dtype]

        self._stream = torch.cuda.Stream(device=self._device)
        self._output = torch.empty(
            self._output_shape,
            dtype=self._output_dtype,
            device=self._device,
        )

        if not self._context.set_tensor_address(
            self._output_name,
            self._output.data_ptr(),
        ):
            raise RuntimeError("failed to bind TensorRT output")

    def infer(self, input_tensor: Any) -> Any:
        """Enqueue one inference and return the reusable output tensor."""
        if tuple(input_tensor.shape) != self._input_shape:
            raise ValueError(
                f"expected shape {self._input_shape}, "
                f"received {tuple(input_tensor.shape)}"
            )

        if not input_tensor.is_cuda:
            raise ValueError("input tensor must be on CUDA")

        if input_tensor.device != self._output.device:
            raise ValueError(
                f"expected device {self._output.device}, "
                f"received {input_tensor.device}"
            )

        if input_tensor.dtype != self._input_dtype:
            raise ValueError(
                f"expected dtype {self._input_dtype}, "
                f"received {input_tensor.dtype}"
            )

        if not input_tensor.is_contiguous():
            raise ValueError("input tensor must be contiguous")

        caller_stream = self._torch.cuda.current_stream(
            input_tensor.device
        )

        self._stream.wait_stream(caller_stream)

        if not self._context.set_tensor_address(
            self._input_name,
            input_tensor.data_ptr(),
        ):
            raise RuntimeError("failed to bind TensorRT input")

        if not self._context.execute_async_v3(
            self._stream.cuda_stream
        ):
            raise RuntimeError("TensorRT execution failed")

        caller_stream.wait_stream(self._stream)
        return self._output

    def synchronize(self) -> None:
        """Wait until this backend's queued TensorRT work completes."""
        self._stream.synchronize()

    @property
    def stream(self) -> Any:
        """Return the CUDA stream used for TensorRT execution."""
        return self._stream
