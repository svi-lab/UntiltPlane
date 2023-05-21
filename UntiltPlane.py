import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import color, morphology, segmentation, exposure, filters
from scipy import optimize


class UntiltPlane(object):
    """Level the plane if the sample is tilted.

    Left click on the image to select the points which will be used to create
    the offset plane. On right-click, this plane will be substracted from
    your image's original data, and the image will be redrawn with updated values.
    Attention: The resulting data will always have zero as its' minimal value.
    That is because after substracting the plane, we also substract the minimum
    of that result.
    Important Note: There is a strong assumption that the offset plane is linear.

    Parameters:
    -----------
    data: 2D numpy.ndarray
        Your input data. NaNs are fine.
    a, b, c: floats
        The initial parameters of the flat offset plane.
        Defaults: `a = np.nanmin(data)`, `b = 0`, `c = 0`
    ftol: float
        What precision of the fit is satisfactory? Default is 1e-10
    **kwargs:
        Arguments passed on to matplotlib.figure

    Output:
    ----------
    corrected_data: numpy.ndarray
        The updated ("untilted") values of your input array.
    plane_params: tuple(float, float, float)
        The tuple of fitted plane parameters (a, b, c). You can also recover them
        as `a`, `b` and `c`



    Example:
    ---------
    ```
    # You just need to run something like:
    >>>my_untilter = UntiltPlane(data)
    # Then, if you want to recover the corrected data:
    >>>corrected_data = my_untilter.corrected_data
    # If you wish to reconstruct the offset plane:
    >>>a, b, c = my_untilter.plan_params
    >>>offset_plane = np.fromfunction(lambda y, x: a + b*x + c*y, data.shape)
    ```
    """

    def __init__(self, data, a="min", b=0, c=0, ftol=1e-10, **kwargs):

        self.data = data
        if a == "min":
            self.a = np.nanmin(self.data)
        else:
            self.a = a
        self.b = b
        self.c = c
        self.ftol = ftol

        self.base_plane = np.empty_like(self.data)
        self.plane_data = []
        self.my_points = []


        figsize = kwargs.pop("figsize", (12, 8))
        facecolor = kwargs.pop("facecolor", "oldlace")
        self.fig, self.aximg = plt.subplots(figsize=figsize, facecolor=facecolor,
                                            **kwargs)

        self.img = self.aximg.imshow(self.data, cmap="cividis")
        self.aximg.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        self.aximg.set_title("Left-click to assign the points that will define\n"+
                            "the offset plane. Right-click when done.")
        self.cid = self.fig.canvas.mpl_connect("button_press_event",
                                               self.onclick)
        plt.show();

    def flat_plane_z(self, y, x):
        return self.a + self.b*x + self.c*y

    def optimize_flat_plane_z(self, independent_coords, a, b, c):
        x = independent_coords[:, 0]
        y = independent_coords[:, 1]
        self.a = a
        self.b = b
        self.c = c
        return self.flat_plane_z(y, x)

    def onclick(self, event):
        if (event.inaxes == self.aximg) and (event.button == 1):
            x, y = event.xdata, event.ydata
            x_ind, y_ind = int(np.round(x)), int(np.round(y))
            z = self.data[y_ind, x_ind]
            if np.isnan(z):
                print(f"You chose a NaN value @(x:{int(x)}, y:{int(y)}), it will be disregarded")
            else:
                self.plane_data.append([x, y, z])
                self.my_points.append(self.aximg.plot(x, y, 'ro', ms=4)[0])
                self.fig.canvas.draw_idle()
        elif event.button != 1:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.detilt()

    def detilt(self):
        """Remove the points, fit the (linear) plane, and update the image"""

        # Remove all the points:
        for point in self.my_points:
            point.remove()
        self.my_points = []

        clicked_data = np.array(self.plane_data)
        independent_coords = clicked_data[:, :2]
        measured_z = clicked_data[:, 2]
        raw_min = np.nanmin(self.data)

        initial_guess = self.a, self.b, self.c

        self.plane_params, self.p_cov = optimize.curve_fit(self.optimize_flat_plane_z,
                                                           independent_coords,
                                                           measured_z,
                                                           initial_guess,
                                                           method="lm",
                                                           ftol=self.ftol)


        self.a, self.b, self.c = self.plane_params
        base_plane = np.fromfunction(self.flat_plane_z, self.data.shape)
        self.corrected_data = self.data - base_plane
        # Correct the offset (not sure whether we should do this or not)
        self.corrected_data -= np.nanmin(self.corrected_data)

        im_min, im_max = 0, np.nanmax(self.corrected_data)
        gs = gridspec.GridSpec(1,2)
        self.aximg.set_position(gs[0].get_position(self.fig))
        self.aximg.set_subplotspec(gs[0])
        self.axcorr = self.fig.add_subplot(gs[1])
        self.axcorr.imshow(self.corrected_data, cmap="cividis")
        self.axcorr.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        self.fig.tight_layout()
        # self.img.set_data(self.corrected_data)
        # self.img.set_clim(im_min, im_max)
        # plt.colorbar(self.img, ax=self.aximg)
        self.axcorr.set_title("You can recover the corrected data with\n`my_untilter.corrected_data`")
        self.fig.canvas.draw_idle()
