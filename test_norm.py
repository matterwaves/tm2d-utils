import tm2d
import numpy as np
from matplotlib import pyplot as plt

pix_size = 1.0

template_atomic = tm2d.TemplateAtomic(
    (8, 8),
    -(pix_size / 2) * np.ones(shape=(1, 3), dtype='float32'),
    disable_convolution=True,
    disable_sigma_e=True
)

ta = template_atomic.make_template(
    rotations = np.array([[0, 0, 0]]),
    pixel_size = pix_size
)
im = ta.read_real(0)[0]

print(im.shape)

plt.imshow(im)
plt.colorbar()
plt.savefig("test_norm.png")