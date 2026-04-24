import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abbreviations import *

@vd.reduce.map_reduce(vd.reduce.SubgroupAdd)
def calc_sums(wave: Buff[v2]) -> v2:
    ind = vc.mapping_index()

    result = vc.new_vec2()

    result.x = wave[ind].x
    result.y = result.x * result.x

    wave[ind].x = result.x
    wave[ind].y = 0

    return result

@vd.shader(exec_size=lambda args: args.image.size)
def apply_normalization(image: Buff[v2], sum_buff: Buff[v2]):
    ind = vc.global_invocation_id().x.to_register()

    sum_vec = (sum_buff[0] / (image.shape.x * image.shape.y)).to_register()
    sum_vec.y = vc.sqrt(sum_vec.y - sum_vec.x * sum_vec.x)

    image[ind].x = (image[ind].x - sum_vec.x) / sum_vec.y

def normalize_signal(signal: vd.Buffer):
    sum_buff = calc_sums(signal) # The reduction returns a buffer with the result in the first value
    apply_normalization(signal, sum_buff)

