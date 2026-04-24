class ParticleStack:
    def __init__(self, image_list=None, pose0_list=None, defocus0_list=None,
                pixel_size=None, is_sim=False):
        self.image = [] if image_list is None else list(image_list)
        self.pose_guess = [] if pose0_list is None else list(pose0_list)
        self.defocus_guess = [] if defocus0_list is None else list(defocus0_list)
        self.pixel_size = pixel_size
        self.is_sim = is_sim

        if self.is_sim:
            self.pose_true = list(self.pose_guess)
            self.defocus_true = list(self.defocus_guess)

    def add_particle(self, im, pose, defocus):
        self.image.append(im)
        self.pose_guess.append(pose)
        self.defocus_guess.append(defocus)
        if self.is_sim:
            self.pose_true.append(pose)
            self.defocus_true.append(defocus)