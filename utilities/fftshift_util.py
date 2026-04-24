import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

@vd.shader(exec_size=lambda args: args.output.size)
def fftshift(output: Buff[c64], input: Buff[c64]):
    ind = vc.global_invocation_id().x.to_dtype(vd.int32).to_register()

    out_x = (ind // output.shape.y).to_register()
    out_y = (ind % output.shape.y).to_register()

    in_x = ((out_x + input.shape.x // 2) % output.shape.x).to_register()
    in_y = ((out_y + input.shape.y // 2) % output.shape.y).to_register()

    output[ind] = input[vc.unravel_index(vc.new_uvec2_register(in_x, in_y), input.shape)]