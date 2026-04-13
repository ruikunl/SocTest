#include <jni.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ncnn/cpu.h"
#include "ncnn/gpu.h"
#include "ncnn/net.h"

namespace {

constexpr int BACKEND_GPU = 1;
constexpr int BACKEND_NNAPI = 2;

struct NcnnSession {
    ncnn::Net net;
    bool use_vulkan = false;
};

jclass runtime_exception_class(JNIEnv* env) {
    return env->FindClass("java/lang/RuntimeException");
}

void throw_runtime(JNIEnv* env, const std::string& message) {
    jclass cls = runtime_exception_class(env);
    if (cls) {
        env->ThrowNew(cls, message.c_str());
    }
}

std::string jstring_to_string(JNIEnv* env, jstring value) {
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars ? chars : "");
    if (chars) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return result;
}

NcnnSession* from_handle(jlong handle) {
    return reinterpret_cast<NcnnSession*>(handle);
}

ncnn::Mat mat_from_float_array(JNIEnv* env, jfloatArray input_array, int width, int height, int channels) {
    ncnn::Mat input(width, height, channels);
    const jsize expected = static_cast<jsize>(width * height * channels);
    const jsize actual = env->GetArrayLength(input_array);
    if (actual < expected) {
        throw_runtime(env, "Input tensor is smaller than expected.");
        return input;
    }

    std::vector<float> values(expected);
    env->GetFloatArrayRegion(input_array, 0, expected, values.data());

    const int plane_size = width * height;
    for (int c = 0; c < channels; ++c) {
        float* channel_ptr = input.channel(c);
        std::copy(
            values.data() + c * plane_size,
            values.data() + (c + 1) * plane_size,
            channel_ptr
        );
    }
    return input;
}

jfloatArray mat_to_float_array(JNIEnv* env, const ncnn::Mat& mat) {
    const int total = static_cast<int>(mat.total());
    jfloatArray result = env->NewFloatArray(total);
    if (!result) {
        return nullptr;
    }

    std::vector<float> values(total);
    const int channels = mat.c > 0 ? mat.c : 1;
    const int plane_size = mat.w * mat.h * mat.elempack;
    int offset = 0;
    for (int c = 0; c < channels; ++c) {
        const float* channel_ptr = mat.channel(c);
        std::copy(channel_ptr, channel_ptr + plane_size, values.data() + offset);
        offset += plane_size;
    }

    env->SetFloatArrayRegion(result, 0, total, values.data());
    return result;
}

jobjectArray make_float_array_pair(JNIEnv* env, jfloatArray first, jfloatArray second) {
    jclass float_array_class = env->FindClass("[F");
    jobjectArray result = env->NewObjectArray(2, float_array_class, nullptr);
    env->SetObjectArrayElement(result, 0, first);
    env->SetObjectArrayElement(result, 1, second);
    return result;
}

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_initializeGpu(JNIEnv*, jobject) {
    ncnn::create_gpu_instance();
}

extern "C" JNIEXPORT void JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_releaseGpu(JNIEnv*, jobject) {
    ncnn::destroy_gpu_instance();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_gpuCount(JNIEnv*, jobject) {
    return ncnn::get_gpu_count();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_nativeCreate(
    JNIEnv* env,
    jobject,
    jstring param_path,
    jstring bin_path,
    jint backend,
    jint threads
) {
    auto session = std::make_unique<NcnnSession>();
    session->use_vulkan = backend == BACKEND_GPU && ncnn::get_gpu_count() > 0;

    const int num_threads = session->use_vulkan ? std::max(1, static_cast<int>(threads)) : 1;
    session->net.opt.num_threads = num_threads;
    ncnn::set_omp_num_threads(num_threads);
    ncnn::set_omp_dynamic(0);
    ncnn::set_kmp_blocktime(0);
    session->net.opt.use_vulkan_compute = session->use_vulkan;
    // Keep the Vulkan path numerically conservative for pose. The more aggressive
    // GPU packing / fp16 paths were fast but produced visibly wrong keypoints.
    session->net.opt.use_packing_layout = false;
    session->net.opt.use_fp16_packed = false;
    session->net.opt.use_fp16_storage = false;
    session->net.opt.use_fp16_arithmetic = false;
    session->net.opt.use_fp16_uniform = false;
    session->net.opt.use_winograd_convolution = false;
    session->net.opt.use_sgemm_convolution = false;
    session->net.opt.use_subgroup_ops = false;
    session->net.opt.use_shader_local_memory = false;

    if (backend == BACKEND_NNAPI) {
        session->net.opt.use_vulkan_compute = false;
        session->use_vulkan = false;
    }

    const std::string param = jstring_to_string(env, param_path);
    const std::string bin = jstring_to_string(env, bin_path);

    if (session->net.load_param(param.c_str()) != 0) {
        throw_runtime(env, "Failed to load ncnn param file: " + param);
        return 0;
    }
    if (session->net.load_model(bin.c_str()) != 0) {
        throw_runtime(env, "Failed to load ncnn model file: " + bin);
        return 0;
    }

    return reinterpret_cast<jlong>(session.release());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_nativeRunPose(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray input_array,
    jint width,
    jint height,
    jint channels
) {
    NcnnSession* session = from_handle(handle);
    if (!session) {
        throw_runtime(env, "ncnn pose session is closed.");
        return nullptr;
    }

    ncnn::Mat input = mat_from_float_array(env, input_array, width, height, channels);
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    ncnn::Extractor extractor = session->net.create_extractor();
    extractor.input("input", input);

    ncnn::Mat simcc_x;
    ncnn::Mat simcc_y;
    if (extractor.extract("simcc_x", simcc_x) != 0) {
        throw_runtime(env, "Failed to extract ncnn output: simcc_x");
        return nullptr;
    }
    if (extractor.extract("simcc_y", simcc_y) != 0) {
        throw_runtime(env, "Failed to extract ncnn output: simcc_y");
        return nullptr;
    }

    jfloatArray x = mat_to_float_array(env, simcc_x);
    jfloatArray y = mat_to_float_array(env, simcc_y);
    return make_float_array_pair(env, x, y);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_nativeRunSegmentation(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray input_array,
    jint width,
    jint height,
    jint channels
) {
    NcnnSession* session = from_handle(handle);
    if (!session) {
        throw_runtime(env, "ncnn segmentation session is closed.");
        return nullptr;
    }

    ncnn::Mat input = mat_from_float_array(env, input_array, width, height, channels);
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    ncnn::Extractor extractor = session->net.create_extractor();
    extractor.input("image", input);

    ncnn::Mat mask;
    if (extractor.extract("mask", mask) != 0) {
        throw_runtime(env, "Failed to extract ncnn output: mask");
        return nullptr;
    }
    return mat_to_float_array(env, mask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_rockenmini_socbenchmark_benchmark_NcnnBridge_nativeClose(JNIEnv*, jobject, jlong handle) {
    delete from_handle(handle);
}
