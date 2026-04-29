import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np
import tm2d

def apply_ctf_to_rfft_buffer(buffer: vd.RFFTBuffer, ctf_params: tm2d.CTFParams, pixel_size: float):
    with vc.shader_context() as ctx:
        shader_args = ctx.declare_input_arguments([vc.Buff[vc.c64]] + ctf_params.get_type_list(1))

        buff = shader_args[0]
        shader_ctf_params = ctf_params.assemble_params_list_from_args(
            shader_args[1:], 1
        )[0]

        ind = vc.global_invocation_id().x.to_register()

        upos_2d = vc.new_uvec2_register()
        upos_2d.x = ind % buffer.shape[2]
        upos_2d.y = ((ind // buffer.shape[2]) + buffer.shape[1] // 2) % buffer.shape[1]

        pos_2d = upos_2d.to_dtype(vc.v2).to_register()
        pos_2d.y = pos_2d.y - buffer.shape[1] // 2

        ctf = tm2d.ctf_filter(
            buffer.shape[1:],
            pos_2d,
            shader_ctf_params,
            pixel_size
        )

        buff[ind] = vc.mult_complex(buff[ind], ctf)

        ctf_apply_shader = vd.make_shader_function(
            description=ctx.get_description("apply_ctf_to_rfft_buffer"),
            exec_count=buffer.size
        )


    ctf_apply_shader(buffer, *ctf_params.get_args(None, 1))

def rfft2_to_fft2(rfft_result, original_shape):
    rows, cols = original_shape
    full_fft = np.zeros((rows, cols), dtype=complex)
    full_fft[:, :rfft_result.shape[1]] = rfft_result

    if cols % 2 == 0:
        for k in range(1, rfft_result.shape[1] - 1):
            full_fft[:, cols - k] = np.conj(np.roll(rfft_result[:, k], 0, axis=0)[::-1])
    else:
        for k in range(1, rfft_result.shape[1]):
            full_fft[:, cols - k] = np.conj(np.roll(rfft_result[:, k], 0, axis=0)[::-1])

    for j in range(1, cols):
        if j < rfft_result.shape[1]:
            continue
        full_fft[1:, j] = np.conj(full_fft[1:, cols - j][::-1])
        full_fft[0, j] = np.conj(full_fft[0, cols - j])

    return full_fft

def generate_ctf(box_size: tuple[int, int], pixel_size: float, ctf_params: tm2d.CTFParams = None) -> np.ndarray:
    result_buffer = vd.RFFTBuffer((1, *box_size))

    ones = np.ones(shape=result_buffer.shape, dtype=np.float32)
    result_buffer.write_fourier((ones).astype(np.complex64))

    apply_ctf_to_rfft_buffer(
        result_buffer,
        ctf_params if ctf_params is not None else tm2d.CTFParams(),
        pixel_size
    )

    rctf2 = result_buffer.read_fourier(0)[0]
    return np.fft.fftshift(rfft2_to_fft2(rctf2, box_size)) / 2 # division due to definition of ctf