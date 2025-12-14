import numpy as np
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray

## ==========================================================================================

def get_polar_grid(h, w):

    # Get grid for cosine ramp function
    h2 = h//2
    w2 = w//2

    # Get normalized frequencies (same as fftfreq) [-1, 1)
    # modulus remainders to account for odd numbers
    wx, wy = np.meshgrid(np.arange(-w2, w2 + (w % 2))/w2, 
                         np.arange(-h2, h2 + (h % 2))/h2)

    # angular component
    angle = np.arctan2(wy, wx)

    # radial component
    radius = np.sqrt(wx**2 + wy**2)
    radius[h2][w2] = radius[h2][w2 - 1] # remove zero component

    return angle, radius

## ==========================================================================================
## Filter Crop functions

def get_filter_crops(filter_in):

    h, w = filter_in.shape
    above_zero = filter_in > 1e-10

    # rows
    dim1 = np.sum(above_zero, axis=1)
    dim1 = np.where(dim1 > 0)[0]
    row_idx = np.clip([dim1.min() - 1, dim1.max() + 1], 0, h)

    # cols
    dim2 = np.sum(above_zero, axis=0)
    dim2 = np.where(dim2 > 0)[0]
    col_idx = np.clip([dim2.min() - 1, dim2.max() + 1], 0, w)

    return np.concatenate((row_idx, col_idx))


def get_cropped_filters(filters, crops):

    cropped_filters = []
    for (filt, crop) in zip(filters, crops):
        cropped_filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])

    return cropped_filters

## ==========================================================================================
## Pyramid Level functions

def build_level(image_dft, filt):

    return np.fft.ifft2(np.fft.ifftshift(image_dft * filt))

def recon_level(pyr_level, filt):
    
    return 2.0 * np.fft.fftshift(np.fft.fft2(pyr_level)) * filt

def build_level_batch(image_dft, filt):

    return np.fft.ifft2(np.fft.ifftshift(image_dft * filt, dim=(1,2)), dim=(1,2))

def recon_level_batch(pyr_level, filt):

    return 2.0 * np.fft.fftshift(np.fft.fft2(pyr_level, dim=(1,2)), dim=(1,2)) * filt



class SteerablePyramid():
    def __init__(self, depth, orientations, filters_per_octave=1, twidth=1, complex_pyr=False):

        # max_depth = int(np.floor(np.log2(np.min(np.array(image.shape)))) - 2)
        self.depth = depth  
        self.orientations = orientations
        self.twidth = twidth
        self.complex_pyr = complex_pyr 

        # number of filters in each band (does not include hi and lo pass)
        self.num_filts = depth*filters_per_octave

        # octaves per filter (bandwidth in terms of octaves)
        self.octave_bw = 1.0/filters_per_octave 


    def _get_radial_mask(self, radius, r):

        # shift log radius (shifts by an octave if log2(r) = 1)
        log_rad = np.log2(radius) - np.log2(r)
        
        hi_mask = np.clip(log_rad, -self.twidth, 0)
        hi_mask = np.abs(np.cos(hi_mask*np.pi/(2*self.twidth)))
        lo_mask = np.sqrt(1.0 - hi_mask**2)

        return lo_mask, hi_mask


    def _get_angle_mask(self, angle, b):
 
        order = self.orientations - 1
        const = np.power(2, (2*order)) * np.power(factorial(order), 2)/(self.orientations*factorial(2*order))
        angle = np.mod(np.pi + angle - np.pi*b/self.orientations, 2*np.pi) - np.pi

        if self.complex_pyr:
            # complex (only use single lobe due to conjugate symmetry)
            angle_mask = 2*np.sqrt(const) * np.power(np.cos(angle), order) * (np.abs(angle) < np.pi/2)
        else:
            # non-complex take magnitude to ensure both lobes are acquired
            angle_mask = np.abs(2*np.sqrt(const) * np.power(np.cos(angle), order))

        return angle_mask


    def get_filters(self, h, w, cropped=False):

        angle, radius = get_polar_grid(h, w)

        # radial_vals specify radial spacing between adjacent filters
        # they determine the lo/hi cutoffs
        radial_vals = 2.0**np.arange(-self.depth, self.octave_bw, self.octave_bw)[::-1]

        # get initial Low and High Pass Filters
        lo_mask_prev, hi_mask = self._get_radial_mask(radius, r=radial_vals[0])

        # get initial crop index
        crop = get_filter_crops(hi_mask)
        crops = [crop]

        if cropped:
            filters = [hi_mask[crop[0]:crop[1], crop[2]:crop[3]]]
        else:
            filters = [hi_mask]

        for idx, rval in enumerate(radial_vals[1:]):
            
            # obtain Radial Band Filter Mask
            lo_mask, hi_mask = self._get_radial_mask(radius, rval)
            rad_mask = hi_mask * lo_mask_prev

            # obtain crops indexes for current level
            if idx > 0:
                crop = get_filter_crops(rad_mask)

            # get filters at each band (orientation)
            for b in range(self.orientations):
                # get Anglular Filter Mask
                angle_mask = self._get_angle_mask(angle, b)
                
                filt = rad_mask*angle_mask/2

                if cropped:
                    filters.append(filt[crop[0]:crop[1], crop[2]:crop[3]])
                else:
                    filters.append(filt) 

                # store crop dimensions for current Pyramid Level
                crops.append(crop)

            lo_mask_prev = lo_mask

        # get final Low Pass Filter Mask and crop dims
        crop = get_filter_crops(lo_mask)
        crops.append(crop)

        if cropped:
            filters.append(lo_mask[crop[0]:crop[1], crop[2]:crop[3]])
        else:
            filters.append(lo_mask)

        return filters, crops
    

    def build_pyramid(self, image, cropped_filters, crops, freq=False):

        image_dft = np.fft.fftshift(np.fft.fft2(image))

        pyramid = []
        for filt, crop in zip(cropped_filters, crops):
            # get filtered/decomposed DFT 
            dft = image_dft[crop[0]:crop[1], crop[2]:crop[3]] * filt

            if freq:
                pyramid.append(dft)
            elif self.complex_pyr:
                pyramid.append(np.fft.ifft2(np.fft.ifftshift(dft)))
            else:
                pyramid.append(np.fft.ifft2(np.fft.ifftshift(dft)).real)


        return pyramid
    

    def build_pyramid_full(self, image, filters, freq=False):
        
        image_dft = np.fft.fftshift(np.fft.fft2(image))[None, :, :]
        dft = image_dft * filters

        if freq:
            return dft
        
        if self.complex_pyr:
            pyramid = np.fft.ifft2(np.fft.ifftshift(dft, axes=(1,2)))
        else:
            pyramid = np.fft.ifft2(np.fft.ifftshift(dft, axes=(1,2))).real
        
        return pyramid


    def reconstruct_image_dft(self, pyramid, cropped_filters, crops, freq=False):

        h, w = pyramid[0].shape
        recon_dft = np.zeros((h, w), dtype=np.complex128)
        for i, (pyr, filt, crop) in enumerate(zip(pyramid, cropped_filters, crops)):
            # dft of sub band
            if freq:
                dft = pyr
            else:
                dft = np.fft.fftshift(np.fft.fft2(pyr))

            # accumulate reconstructed sub bands
            if self.complex_pyr and (i !=0 ) and (i != (len(cropped_filters) - 1)):
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += 2.0*dft*filt
            else:
                recon_dft[crop[0]:crop[1], crop[2]:crop[3]] += dft*filt

        return recon_dft
    

    def reconstruct_image_dft_full(self, pyramid, filters, freq=False):

        h, w = pyramid[0].shape
        recon_dft = np.zeros((h, w), dtype=np.complex128)
        for i, (pyr, filt) in enumerate(zip(pyramid, filters)):
            # dft of sub band
            if freq:
                dft = pyr
            else:
                dft = np.fft.fftshift(np.fft.fft2(pyr))

            # accumulate reconstructed sub bands
            if self.complex_pyr and (i !=0 ) and (i != (len(filters) - 1)):
                recon_dft += 2.0*dft*filt
            else:
                recon_dft += dft*filt

        return recon_dft
    

    def reconstruct_image(self, pyramid, filters, crops=None, full=False, freq=False):

        if full:
            recon_dft = self.reconstruct_image_dft_full(pyramid, filters, freq)
        else:
            recon_dft = self.reconstruct_image_dft(pyramid, filters, crops, freq)

        return np.fft.ifft2(np.fft.ifftshift(recon_dft)).real


    def displayfil(self, filters, title="filters"):

        fig, ax = plt.subplots(self.num_filts, self.orientations, figsize=(30, 20))
        fig.suptitle(title, size=22)

        idx = 0
        for i in range(self.num_filts):
            idx = i*self.orientations
            for j in range(1, self.orientations + 1):
                jdx = idx + j
                ax[i][j - 1].imshow(filters[jdx])

        plt.tight_layout();

        return fig, ax
    
    def displaypyr(self, pyramid, title="pyramid"):

        fig, ax = plt.subplots(self.num_filts, self.orientations, figsize=(30, 20))
        fig.suptitle(title, size=22)

        idx = 1
        for i in range(self.num_filts):
            idx = i*self.orientations
            for j in range(1, self.orientations + 1):
                jdx = idx + j
                ax[i][j - 1].imshow(np.abs(pyramid[jdx]))

        plt.tight_layout();

        return fig, ax
    
img = rgb2gray(data.astronaut())
h, w = img.shape
SP = SteerablePyramid(3, 8, filters_per_octave=1, twidth=1, complex_pyr=True)
filters, crops = SP.get_filters(h, w, cropped=False)
pyramid = SP.build_pyramid_full(img, filters, freq=False)
reconstructed = SP.reconstruct_image(pyramid, filters, crops=crops, full=True, freq=False)
#plt.imshow(reconstructed)
#SP.displaypyr(pyramid)

