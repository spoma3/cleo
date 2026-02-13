import numpy as np
from scipy.optimize import curve_fit
from attrs import define, field
import matplotlib.pyplot as plt
import json
import os
import math

@define(eq=False)
class ExcitationModel:
    """Base class for excitation models."""
    pass

# @define(eq=False)
# class LightExcitation(ExcitationModel):
#     """Models light-dependent excitation"""

#     # model: str = field(default="exc_factor = some_function(Irr_pre) : 1", init=False)
#     """Uses a Hill equation to convert from Ca2+ to ΔF/F, as in Song et al., 2021"""
#     name: str = field(default='GECI', kw_only=True)
#     A: float = field(kw_only=True)
#     """The amplitude of Hill equation as the maximum amount of fluorescence at a given brightness level."""
#     baseline: float = field(kw_only=True)
#     """The baseline of the Hill equation at any given light intensity."""
#     k: Quantity = field(kw_only=True)
#     """the decay parameter of the model how quickly does normalized fluorescence decay as the brightness level increases past its maximum saturation level."""
#     # x: float = field(kw_only=True)
#     """the brightness of the function for the hills exponential decay equation."""
#     n: float = field(kw_only=True)
#     """the steepness of the transition with higher values meaning a more sudden
#       more switch like response and a lower value meaning a smoother more gradual response function."""
#     ec50: float = field(kw_only=True)
#     """the value of the brightness where the response is halfway between the bottom and Fmax"""
#     #Multivariate log normal distribution of wavelength to fluorescence
#     baseline_w: float = field(kw_only=True)
#     """baseline of the log normal equation when the wavelength is 0."""
#     # w: Quantity = field(kw_only=True)
#     """the wavelength of the function for the log normal equation."""
#     A1: Quantity = field(kw_only=True)
#     """the amplitude of the first log normal distribution highlighting the maximum height of that first log normal function."""
#     sigma1: Quantity = field(kw_only=True)
#     """the standard distribution or the width of the first log normal distribution highlighting how wide the curve is."""
#     A2: Quantity = field(kw_only=True)
#     """the amplitude of the second log normal distribution highlighting the maximum height of that second log normal function."""
#     sigma2: Quantity = field(kw_only=True)
#     """the standard distribution or the width of the second log normal distribution highlighting how wide the curve is."""
#     mu1: Quantity = field(kw_only=True)
#     """the mu1 is the mean of the distribution or the center of the first log normal distribution."""
#     mu2: Quantity = field(kw_only=True)
#     """the mu2 is the mean of the distribution or the center of the second log normal distribution."""
#     # dF_w = baseline_w + ((A1 * (1/(lam/nmeter * sigma1 * (2 * pi)**0.5)) * exp(-(log(lam/nmeter) - mu1)**2/(2 * sigma1**2)))) + ((A2 * (1/(lam/nmeter * sigma2 * (2 * pi)**0.5)) * exp(-(log(lam/nmeter) - mu2)**2/(2 * sigma2**2)))) : 1
#     model: str = field(
#         default = """
#         dF_p = baseline + ((A) * (phi * (1/(1/meter**2/second)))**n)/(ec50**n + (phi * (1/(1/meter**2/second)))**n) * exp(-k * phi * (1/(1/meter**2/second))) : 1
#         dF_w = baseline_w + ((A1 * (1/(lam_safe * sigma1 * (2 * pi)**0.5)) * exp(-(log(lam_safe) - mu1)**2/(2 * sigma1**2)))) + ((A2 * (1/(lam_safe * sigma2 * (2 * pi)**0.5)) * exp(-(log(lam_safe) - mu2)**2/(2 * sigma2**2)))) : 1
#         lam_safe = clip(lam/nmeter, 1e-9, inf): 1
#         lam : meter (shared)
#         exc_factor = dF_p * dF_w : 1
#         """,
#         init=False,
#     )
#     def __attrs_post_init__(self):
#         self.model = self.model

@define(eq=False)
class LightExcitation(ExcitationModel):
    """Models light-dependent excitation using a Hill function"""

    A: float = field(init=False)
    Kd: float = field(init=False)
    n: float = field(init=False)

    def hill_function_decay(self, Irr_pre, A, Kd, n, k, x):
        return A * (Irr_pre ** n) / ((Kd ** n) + (Irr_pre ** n)) * np.exp(-k * x)

    def fit_excitation(self, light_intensities, responses):
        # Fit Hill function to the light intensity vs response data
        popt, _ = curve_fit(self.hill_function_decay, light_intensities, responses, maxfev=10000)
        self.A, self.Kd, self.n = popt

    def __call__(self, Irr_pre):
        return self.hill_function_decay(Irr_pre, self.A, self.Kd, self.n, self.k, self.x)

    model: str = field(default="exc_factor = hill_function(Irr_pre) : 1", init=False)

@define(eq=False)
class GECI:
    """Base class for GECIs"""
    light_dependent: bool = field(default=False)

@define(eq=False)
class LightDependent:
    """Base class for light-dependent behavior"""
    pass

@define(eq=False)
class LightDependentGECI(GECI, LightDependent):
    """Light-dependent calcium indicator"""
    exc_model: LightExcitation = field(kw_only=True)

def gcamp6f(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def gcamp6s(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp3(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def ogb1(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def gcamp6rs09(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def gcamp6rs06(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp7f(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp7s(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp7b(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp7c(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp8s(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp8m(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp8f(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def xcampgf(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp3(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp5d(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp5a(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp5g(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def jgcamp2(light_dependent=False, exc_model=None):
    return LightDependentGECI(light_dependent=light_dependent, exc_model=exc_model if light_dependent else None)

def fit_light_dependence_curves(indicators, light_intensities, responses_dict):
    """
    Fit light dependence curves for multiple indicators.

    Parameters
    ----------
    indicators : list
        List of indicator functions (e.g., [gcamp6f, gcamp6s, gcamp3, ...]).
    light_intensities : np.ndarray
        Array of light intensity values.
    responses_dict : dict
        Dictionary where keys are indicator names and values are arrays of responses to light intensities.
    """
    fitted_indicators = {}
    plt.figure(figsize=(10, 8))

    for i, indicator_fn in enumerate(indicators):
        indicator_name = indicator_fn.__name__
        if indicator_name in responses_dict:
            responses = responses_dict[indicator_name]
            exc_model = LightExcitation()
            geci_model = indicator_fn(light_dependent=True, exc_model=exc_model)
            geci_model.exc_model.fit_excitation(light_intensities, responses)
            fitted_indicators[indicator_name] = geci_model
            print(f"Fitted {indicator_name}: A={geci_model.exc_model.A}, Kd={geci_model.exc_model.Kd}, n={geci_model.exc_model.n}")

            # Plotting the results
            plt.subplot(3, 4, i + 1)
            plt.scatter(light_intensities, responses, label='Data')
            fitted_responses = geci_model.exc_model(light_intensities)
            plt.plot(light_intensities, fitted_responses, label='Fitted curve', color='red')
            plt.title(indicator_name)
            plt.xlabel('Light Intensity')
            plt.ylabel('Response')
            plt.legend()
        else:
            print(f"No response data for {indicator_name}")

    plt.tight_layout()
    plt.show()
    return fitted_indicators

# Example usage
indicators = [gcamp6f, gcamp6s, jgcamp3, ogb1, gcamp6rs09, gcamp6rs06, jgcamp7f, jgcamp7s, jgcamp7b, jgcamp7c,
               jgcamp8s, jgcamp8f, jgcamp8m, xcampgf]
light_intensities = np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30])  # Example light intensities
light_intensities_dict = {
    "jgcamp3" : np.array([1.0, 1.5, 1.97, 3.92, 2.46, 5.0, 5.9, 7.86, 9.76, 12.69, 16.76, 19.70, 24.38, 29.10, 38.24, 49.10, 56.55, 75.95]),
    "jgcamp7c" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp7s" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp7b" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp7f" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp6f" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp6s" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),

    "jgcamp8s" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp8f" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "jgcamp8m" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),
    "xcampgf" : np.array([2, 5, 7.5, 10, 12.5, 15, 20, 25, 30]),

    "jgcamp2" : np.array([1.0, 1.5, 1.9, 1.97, 2.48, 2.76, 2.96, 3.67, 4.64, 5.74, 7.65, 12.50, 9.70, 19.42, 16.63, 28.94, 24.14, 38.90, 47.91, 57.53, 76.46]),
    "jgcamp5a" : np.array([0.99, 1.0, 1.5, 1.9, 2.04, 2.50, 2.88, 3.03, 3.80, 4.83, 8.0, 16.86, 20.14, 25.10, 30.26, 40.41, 80.19]),
    "jgcamp5g" : np.array([0.99, 1.02, 1.51, 1.91, 2.53, 2.0, 3.03, 2.86, 4.81, 2.49, 2.88, 3.82, 3.012, 7.89, 25.47, 5.92, 20.22, 10.08, 40.08, 50.69, 80.19]),
    "jgcamp5d" : np.array([2.85, 3.94, 4.77, 6.00, 7.89, 10.00, 13.06, 17.10, 20.38, 25.25, 29.98, 40.55, 50.61, 60.21, 81.50])
}
responses_dict = {
    "jgcamp3": (np.array([0.17, 0.35, 0.67, 0.97, 1.38, 2.44, 3.45, 4.52, 6.67, 8.03, 9.08, 9.47, 9.37, 9.04, 8.91, 7.65, 7.30, 6.20, 5.32]) - 0.17)/0.17,
    "jgcamp7c": (np.array([0.35, 3.451, 7.14, 9.96, 11.61, 12.5, 12.3, 11.6, 10.9]) - 0.35)/0.35,
    "jgcamp7s": (np.array([0.86, 4.3, 8.2, 10.9, 12.2, 12.7, 12.3, 11.4, 10.5]) - 0.86)/0.86,
    "jgcamp7b": (np.array([1, 5, 9, 11.6, 13, 13.2, 12.6, 11.8, 10.2]) - 1)/1,
    "jgcamp7f": (np.array([0.973, 5.23, 8.4, 11.5, 13.1, 14, 14.2, 13.2, 12.5]) - 0.973)/0.973,
    "jgcamp6f": (np.array([1.06, 5.5, 9.8, 12.8, 14.5, 14.9, 14.7, 13.9, 12.8]) - 1.06)/1.06,
    "jgcamp6s": (np.array([0.98, 5.61, 10.2, 13.5, 15.3, 15.9, 15.6, 14.7, 13.8]) - 0.98)/0.98,

    # Add responses for other indicators here
    "jgcamp8s": (np.array([0.83, 4.27, 7.7, 10.07, 11.37, 12.12, 12.00, 11.91, 11.37]) - 0.83)/0.83,
    "jgcamp8f": (np.array([0.81, 4.65, 8.3, 10.83, 12.12, 12.59, 12.47, 12.08, 11.77]) - 0.81)/0.81,
    "jgcamp8m": (np.array([0.795, 4.35, 7.89, 10.39, 11.62, 12.33, 12.14, 11.37, 10.96]) - 0.795)/0.795,
    "xcampgf": (np.array([0.46, 2.91, 5.72, 8.24, 10.14, 11.05, 11.57, 11.65, 11.16]) - 0.46)/0.46,

    "jgcamp2": (np.array([0.16, 0.35, 0.57, 0.63, 0.95, 1.18, 1.36, 1.86, 2.68, 3.52, 4.75, 5.45, 6.15, 6.45, 6.48, 6.36, 5.98, 5.42, 5.15, 4.78, 4.39]) - 0.16)/0.16,
    "jgcamp5a": (np.array([0.19, 0.41, 0.67, 0.72, 1.06, 1.42, 1.54, 2.24, 3.35, 4.46, 6.41, 7.95, 8.94, 9.70, 9.72, 9.60, 9.16, 7.99, 7.15, 6.35, 5.28]) - 0.19)/0.19,
    "jgcamp5g": (np.array([0.20, 0.44, 0.68, 0.79, 1.15, 1.51, 1.71, 2.50, 3.75, 4.67, 6.54, 7.91, 8.89, 8.99, 8.75, 8.28, 7.63, 6.88, 5.05]) - 0.20)/0.20,
    "jgcamp5d": (np.array([0.16, 0.35, 0.58, 1.24, 2.04, 2.89, 4.08, 5.76, 7.10, 8.24, 9.07, 9.16, 9.10, 8.56, 7.79, 7.18, 6.79, 5.83]) - 0.16)/0.16,

}

responses_dict_2 = {
    "jgcamp3": np.array([0.17, 0.36, 0.67, 2.40, 0.98, 3.53, 4.52, 6.76, 8.09, 9.21, 9.59, 9.56, 9.13, 8.99, 7.68, 7.15, 6.32, 5.14])/0.17,
    "jgcamp2": np.array([0.17, 0.36, 0.57, 0.64, 0.96, 1.17, 1.36, 1.87, 2.67, 3.53, 4.85, 6.13, 5.48, 6.47, 6.60, 5.94, 6.42, 5.39, 5.16, 4.82, 4.36])/0.17,
    "jgcamp5a": np.array([1.02, 0.19, 0.19, 0.41, 0.64, 0.72, 1.06, 1.44, 1.54, 2.28, 3.36, 6.18, 9.98, 9.99, 9.51, 9.26, 8.22, 5.15])/1.02,
    "jgcamp5g": np.array([1.02, 0.20, 0.45, 0.69, 1.14, 0.78, 1.67, 1.51, 3.69, 1.13, 1.44, 2.50, 1.65, 6.63, 8.63, 4.67, 8.79, 7.92, 7.36, 6.79, 5.15])/1.02,
    "jgcamp5d": np.array([1.24, 2.03, 2.92, 4.09, 5.68, 7.13, 9.07, 8.99, 9.09, 8.82, 8.43, 7.75, 7.11, 6.72, 5.77])/1.24,
}
# fitted_indicators = fit_light_dependence_curves(indicators, light_intensities, responses_dict)
def hill_function(x, bottom, top, ec50, n):
    return bottom + (top - bottom) * (x**n) / (ec50**n + x**n)

ncols = 3
nrows = math.ceil(len(indicators)/ncols)
fig, axes = plt.subplots(nrows, ncols, figsize = (5 * ncols, 4 * nrows))
axes = axes.flatten()

indic_eq = {}
with open("hills_decay_function.json", "w") as f:
    pass

def hill_decay(x, Fmax, ec50, n, k, bottom=0):
    return bottom + (Fmax * (x**n)/(ec50**n + x**n)) * np.exp(-k * x)
# for light_intensity, indicator in zip(list(responses_dict.keys()), list(light_intensities_dict.keys())):
plt.close('all')
for indicator, responses in responses_dict_2.items():
    if indicator not in light_intensities_dict:
        print("not in the light_intensities_dict.")
        continue
    light_intensities = light_intensities_dict[indicator]
    if (len(light_intensities) != len(responses)):
        print(f"Lengths dont match {len(light_intensities)} and {len(responses)} for {indicator}")
        continue
    export_dir = os.path.join(indicator)
    os.makedirs(export_dir, exist_ok=True)
    p0 = [max(responses), np.median(light_intensities), 2.0, 0.01, min(responses)]
    try:
        popt, pcov = curve_fit(hill_decay, light_intensities, responses, p0=p0,
                               bounds = ([0, 0, 0.1, 0, 0], [np.inf, np.inf, 10, 1, np.inf]), maxfev=5000)
        Fmax, ec50, n, k, bottom = popt
        x_fit = np.linspace(min(light_intensities), max(light_intensities), 200)
        y_fit = hill_decay(x_fit, *popt)
        y_pred = hill_decay(light_intensities, *popt)

        residuals = responses - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((responses - np.mean(responses))**2)
        r_squared = 1 - (ss_res/ss_tot)

        plt.figure()
        plt.scatter(light_intensities, responses, label=f"{indicator} brightness (mW) to fluorescence (dF/F0)")
        plt.plot(x_fit, y_fit, 'r-', label = 'Hill decay fit')
        plt.xlabel('Light Intensities (mW)')
        plt.ylabel('Response (∆F/F0)')
        plt.title(indicator)
        plt.legend()
        plt.text(0.05, 0.95, f"R^2 = {r_squared:.3f}",
                 transform = plt.gca().transAxes, fontsize = 12, verticalalignment = "top",
                 bbox = dict(facecolor="white", alpha=0.7, edgecolor="none"))
        plt.savefig(os.path.join(export_dir, f"{indicator}_fit.png"))
        plt.show()
        print(f"indicator is: {indicator}")
        eq = f"bottom : {bottom}, Fmax : {Fmax}, ec50 : {ec50}, n : {n}, k : {k}"
        with open(os.path.join(export_dir, f"{indicator}_fit.txt"), "w") as f:
            f.write(f"Equation parameters for {indicator}\n")
            f.write(f"{eq}")
        indic_eq[indicator] = {
            'bottom' : bottom,
            'Fmax' : Fmax,
            'ec50' : ec50,
            'n' : n,
            'k' : k,
        }
        print(f"eq is: {eq}")
        print(f"R^2 is: {r_squared}")
    except Exception as e:
        print(f"Skipped because of {e}")
        continue

with open("hills_decay_function.json", "a") as f:
    json.dump(indic_eq, f, indent=4)