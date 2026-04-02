from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, glob

home = os.path.expanduser("~")

current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="nvfp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="scaled_fp4_ops",
            sources=["binding.cpp", *glob.glob("kernel/*.cu")],
            include_dirs=[
                os.path.join(current_dir, "kernel"),
                os.path.join(home, "cutlass", "include"),
                os.path.join(home, "cutlass", "tools/util/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_120a,code=sm_120a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.8.0",
    ],
    python_requires=">=3.8",
)
