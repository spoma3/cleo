from __future__ import annotations

from attrs import asdict, define, field, fields_dict
from brian2 import NeuronGroup, Quantity, Synapses, nmolar, np, second, umolar

from cleo.base import SynapseDevice
from cleo.light import LightDependent
from cleo.utilities import brian_safe_name
from Dictionaries import Light_Intensity_Dict, Wavelength_Dict


@define(eq=False)
class Sensor(SynapseDevice):
    """Base class for sensors"""

    sigma_noise: float = field(kw_only=True)
    """standard deviation of Gaussian noise in ΔF/F measurement"""
    dFF_1AP: float = field(kw_only=True)
    """ΔF/F for 1 AP, only used for scope SNR cutoff"""
    location: str = field(kw_only=True)
    """where sensor is expressed: cytoplasm or membrane"""

    @location.validator
    def _check_location(self, attribute, value):
        if value not in ("cytoplasm", "membrane"):
            raise ValueError(
                f"indicator location must be 'cytoplasm' or 'membrane', not {value}"
            )

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio for 1 AP"""
        return self.dFF_1AP / self.sigma_noise

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        """Returns a {neuron_group: fluorescence} dict of dFF values."""
        pass

    @property
    def exc_spectrum(self) -> list[tuple[float, float]]:
        """Excitation spectrum, alias for :attr:`spectrum`"""
        return self.spectrum


@define(eq=False)
class CalciumModel:
    """Base class for how GECI computes calcium concentration

    Must provide variable Ca (molar) in model."""

    model: str
    on_pre: str = field(default="", init=False)

@define(eq=False)
class PreexistingCalcium(CalciumModel):
    """Calcium concentration is pre-existing in neuron model"""

    model: str = field(default="Ca = Ca_pre : mmolar", init=False)


@define(eq=False)
class DynamicCalcium(CalciumModel):
    """Simulates intracellular calcium dynamics from spikes.
    Pieced together from Lütke et al., 2013; Helmchen and Tank, 2015;
    and Song et al., 2021. (`code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_)
    """

    on_pre: str = field(default="Ca += dCa_T / (1 + kappa_S + kappa_B)", init=False)
    """from eq 9 in Lütke et al., 2013"""
    model: str = field(
        default="""
            dCa/dt = -gamma * (Ca - Ca_rest) / (1 + kappa_S + kappa_B) : mmolar (clock-driven)
            kappa_B = B_T * K_d / (Ca + K_d)**2 : 1""",
        init=False,
    )
    """from eq 8 in Lütke et al., 2013"""

    Ca_rest: Quantity = field(kw_only=True)
    """resting Ca2+ concentration (molar)"""
    gamma: Quantity = field(kw_only=True)
    """clearance/extrusion rate (1/sec)"""
    B_T: Quantity = field(kw_only=True)
    """total indicator (buffer) concentration (molar)"""
    kappa_S: float = field(kw_only=True)
    """Ca2+ binding ratio of the endogenous buffer"""
    dCa_T: Quantity = field(kw_only=True)
    """total Ca2+ concentration increase per spike (molar)"""

    def init_syn_vars(self, syn: Synapses) -> None:
        syn.Ca = self.Ca_rest


@define(eq=False)
class CalBindingActivationModel:
    """Base class for modeling calcium binding/activation"""

    model: str


@define(eq=False)
class NullBindingActivation(CalBindingActivationModel):
    """Doesn't model binding/activation; i.e., goes straight from [Ca2+] to ΔF/F"""

    model: str = field(default="CaB_active = Ca: mmolar", init=False)


@define(eq=False)
class DoubExpCalBindingActivation(CalBindingActivationModel):
    """Double exponential kernel convolution representing CaB binding/activation.

    Convolution is implemented via ODEs; see ``notebooks/double_exp_conv_as_ode.ipynb``
    for derivation.

    :attr:`A`, :attr:`tau_on`, and :attr:`tau_off` are the versions with proper scale and units of NAOMi's
    ``ca_amp``, ``t_on``, and ``t_off``.

    Some parameters found `here <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/check_cal_params.m?at=master#lines-90>`_.
    Fitting code `here <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/MiscCode/fit_NAOMi_calcium.m?at=master#fit_NAOMi_calcium.m>`_.
    """

    model: str = field(
        default="""
            CaB_active = Ca_rest + b : mmolar  # add tiny bit to avoid /0
            db/dt = beta : mmolar (clock-driven)
            lam = 1/tau_off + 1/tau_on : 1/second
            kap = 1/tau_off : 1/second
            dbeta/dt = (                    # should be M/s/s
                A * (lam - kap) * (Ca - Ca_rest)  # M/s/s
                - (kap + lam) * beta        # M/s/s
                - kap * lam * b    # M/s/s
            ) : mmolar/second (clock-driven)
            """,
        init=False,
    )

    A: float = field(kw_only=True)
    """amplitude of double exponential kernel"""
    tau_on: Quantity = field(kw_only=True)
    """CaB binding/activation time constant (sec)"""
    tau_off: Quantity = field(kw_only=True)
    """CaB unbinding/deactivation time constant (sec)"""
    Ca_rest: Quantity = field(kw_only=True)
    """Resting Ca2+ concentration (molar)."""

    def init_syn_vars(self, syn: Synapses) -> None:
        syn.b = 0
        syn.beta = 0


@define(eq=False)
class ExcitationModel:
    """Defines ``exc_factor``"""

    model: str = field(default="exc_factor = 1 : 1", init=False)


@define(eq=False)
class NullExcitation(ExcitationModel):
    """Models excitation as a constant factor"""

    model: str = field(default="exc_factor = 1 : 1", init=False)


@define(eq=False)
class LightExcitation(ExcitationModel):
    """Models light-dependent excitation (not implemented yet)"""

    model: str = field(default="exc_factor = some_function(Irr_pre) : 1", init=False)


@define(eq=False)
class GECI(Sensor):
    """GECI model based on Song et al., 2021, with interchangeable components.

    See :func:`geci` for a convenience function for creating GECI models.

    As a potentially simpler alternative for future work, see the phenomological S2F model
    from `Zhang et al., 2023 <https://www.nature.com/articles/s41586-023-05828-9>`_.
    While parameter count looks similar, at least they have parameters fit already, and directly
    to data, rather than to biophysical processes before the data.
    """

    cal_model: CalciumModel = field(kw_only=True)
    bind_act_model: CalBindingActivationModel = field(kw_only=True)
    exc_model: ExcitationModel = field(kw_only=True)

    """Uses a Hill equation to convert from Ca2+ to ΔF/F, as in Song et al., 2021"""
    K_d: Quantity = field(kw_only=True)
    """indicator dissociation constant (binding affinity) (molar)"""
    n_H: float = field(kw_only=True)
    """Hill coefficient for conversion from Ca2+ to ΔF/F"""
    dFF_max: float = field(kw_only=True)
    """amplitude of Hill equation for conversion from Ca2+ to ΔF/F,
    Fmax/F0. May be approximated from 'dynamic range' in literature Fmax/Fmin"""

    location: str = field(default="cytoplasm", init=False)
    fluor_model: str = field(
        default="""
            dFF_baseline = 1 / (1 + (K_d / Ca_rest) ** n_H) : 1
            dFF = exc_factor * rho_rel * dFF_max  * (
                1 / (1 + (K_d / CaB_active) ** n_H)
                - dFF_baseline
            ) : 1
            rho_rel : 1
        """,
        init=False,
    )

    def get_state(self) -> dict[NeuronGroup, np.ndarray]:
        return {ng_name: syn.dFF for ng_name, syn in self.synapses.items()}

    def __attrs_post_init__(self):
        self.model = "\n".join(
            [
                self.cal_model.model,
                self.bind_act_model.model,
                self.exc_model.model,
                self.fluor_model,
            ]
        )
        self.on_pre = self.cal_model.on_pre

    def init_syn_vars(self, syn: Synapses) -> None:
        for model in [self.cal_model, self.bind_act_model, self.exc_model]:
            if hasattr(model, "init_syn_vars"):
                model.init_syn_vars(syn)

    @property
    def params(self) -> dict:
        """Returns a dictionary of all parameters from model/submodels"""
        params = asdict(self, recurse=False)
        # remove generic fields that are not parameters
        for field in fields_dict(Sensor):
            params.pop(field)
        # remove private attributes
        for key in list(params.keys()):
            if key.startswith("_"):
                params.pop(key)
        # add params from sub-models
        for model in [self.cal_model, self.bind_act_model, self.exc_model]:
            to_add = asdict(model, recurse=False)
            to_add.pop("model")
            params.update(to_add)
        return params

@define(eq=False, slots=False)
class LightDependentGECI(LightDependent):
    """Light-dependent calcium indicator (not yet implemented)"""
    """Uses a Hill equation to convert from Ca2+ to ΔF/F, as in Song et al., 2021"""
    name: str = field(default='GECI', kw_only=True)
    A: float = field(kw_only=True)
    """The amplitude of Hill equation as the maximum amount of fluorescence at a given brightness level."""
    baseline: float = field(kw_only=True)
    """The baseline of the Hill equation at any given light intensity."""
    k: Quantity = field(kw_only=True)
    """the decay parameter of the model how quickly does normalized fluorescence decay as the brightness level increases past its maximum saturation level."""
    x: float = field(kw_only=True)
    """the brightness of the function for the hills exponential decay equation."""
    n: float = field(kw_only=True)
    """the steepness of the transition with higher values meaning a more sudden
      more switch like response and a lower value meaning a smoother more gradual response function."""
    ec50: float = field(kw_only=True)
    """the value of the brightness where the response is halfway between the bottom and Fmax"""
    #Multivariate log normal distribution of wavelength to fluorescence
    baseline_w: float = field(kw_only=True)
    """baseline of the log normal equation when the wavelength is 0."""
    w: Quantity = field(kw_only=True)
    """the wavelength of the function for the log normal equation."""
    A1: Quantity = field(kw_only=True)
    """the amplitude of the first log normal distribution highlighting the maximum height of that first log normal function."""
    sigma1: Quantity = field(kw_only=True)
    """the standard distribution or the width of the first log normal distribution highlighting how wide the curve is."""
    A2: Quantity = field(kw_only=True)
    """the amplitude of the second log normal distribution highlighting the maximum height of that second log normal function."""
    sigma2: Quantity = field(kw_only=True)
    """the standard distribution or the width of the second log normal distribution highlighting how wide the curve is."""
    mu1: Quantity = field(kw_only=True)
    """the mu1 is the mean of the distribution or the center of the first log normal distribution."""
    mu2: Quantity = field(kw_only=True)
    """the mu2 is the mean of the distribution or the center of the second log normal distribution."""
    exc_factor: str = field(
        default = """
        dF_p = baseline + ((A) * x**n)/(ec50**n + x**n) * exp(-k * x) : 1
        dF_w = baseline_w + ((A1 * (1/(w * sigma1 * (2 * pi)**0.5)) * exp(-(log(w) - mu1)**2/(2 * sigma1**2)))) + ((A2 * (1/(w * sigma2 * (2 * pi)**0.5)) * exp(-(log(w) - mu2)**2/(2 * sigma2**2)))) : 1
        dFF = dF_p * dF_w : 1
        """,
        init=False,
    )
    cal_model: CalciumModel = field(kw_only=True)
    bind_act_model: CalBindingActivationModel = field(kw_only=True)
    exc_model: ExcitationModel = field(kw_only=True)
    location: str = field(default="cytoplasm", init=True)
    def __attrs_pre_init__(self):
        if not hasattr(self, "name"):
            self.name = "GECI"
    def __attrs_post_init__(self):
        super_post_init = getattr(super(), "_attrs_post_init_", None)
        if callable(super_post_init):
            super_post_init()

    def fluorescence(self, GECI_str, brightness, wavelength):
        parameters, params_w = LightDependentParams(GECI_str)
        parameters['x'] = brightness
        params_w['w'] = wavelength
        params = {**parameters, **params_w}
        exc_factor = """
        dF_p = baseline + ((A) * x**n)/(ec50**n + x**n) * exp(-k * x) : 1
        dF_w = baseline_w + ((A1 * (1/(w * sigma1 * (2 * pi)**0.5)) * exp(-(log(w) - mu1)**2/(2 * sigma1**2)))) + ((A2 * (1/(w * sigma2 * (2 * pi)**0.5)) * exp(-(log(w) - mu2)**2/(2 * sigma2**2)))) : 1
        dFF = dF_p * dF_w : 1
        """
        ng = NeuronGroup(1, model=exc_factor, method="euler", namespace=params)
        print(f"ng.dFF is: {ng.dFF}")
        return ng.dFF

def LightDependentParams(GECI_str) -> dict:
        params = Light_Intensity_Dict[GECI_str]
        params_w = Wavelength_Dict[GECI_str]
        A = params['A']
        baseline = params['baseline']
        ec50 = params['ec50']
        n = params['n']
        k = params['k']
        return {'A' : A, 'baseline' : baseline, 'k' : k, 'n' : n, 'ec50' : ec50}, params_w

def light_dependent_geci(name: str, doub_exp_conv: bool, pre_existing_cal: bool, bind_act_model: bool,  x, w, Ca_rest=50 * nmolar,
                         gamma=92.3/second, B_T=200*umolar, kappa_S=110, dCa_T=7.6 * umolar):
    params, params_w = LightDependentParams(name)
    print(f"params is: {params}")
    g = geci(light_dependent=True, doub_exp_conv=False, pre_existing_cal=False, K_d=params['ec50'], n_H=params['n'], dFF_max=params['A'],
             Ca_rest=50 * nmolar,
                         gamma=92.3/second, B_T=200*umolar, kappa_S=110, dCa_T=7.6 * umolar, sigma_noise=1.0, dFF_1AP=0.2, x=x, w=w, name=name, **params, **params_w)
    params_kept = {}
    for kw_param_key, kw_param_value in params.items():
        if (kw_param_key == 'bottom'):
            kw_param_key = 'baseline'
        params_kept[kw_param_key] = kw_param_value
    for kw_param_key, kw_param_value in params_w.items():
        if (kw_param_key == 'baseline'):
            kw_param_key = 'baseline_w'
        params_kept[kw_param_key] = kw_param_value
    return LightDependentGECI(cal_model=g.cal_model, exc_model=g.exc_model, bind_act_model=g.bind_act_model,
                               sigma_noise=1.0, dFF_1AP=0.2, location='cytoplasm', **params_kept)

def geci(
    light_dependent: bool, doub_exp_conv: bool, pre_existing_cal: bool, **kwparams
) -> GECI:
    """Initializes a :class:`GECI` model with given parameters.

    Parameters
    ----------
    light_dependent : bool
        Whether the indicator is light-dependent.
    doub_exp_conv : bool
        Whether to use double exponential convolution for binding/activation.
    pre_existing_cal : bool
        Whether to use calcium concentrations already simulated in the neuron model.
    **kwparams
        Keyword parameters for :class:`GECI` and sub-models.

    Returns
    -------
    GECI
        A (LightDependent)GECI model specified submodels and parameters.
    """
    ExcModel = LightExcitation if light_dependent else NullExcitation
    CalModel = PreexistingCalcium if pre_existing_cal else DynamicCalcium
    BAModel = DoubExpCalBindingActivation if doub_exp_conv else NullBindingActivation

    def init_from_kwparams(cls, **more_kwargs):
        kwparams.update(more_kwargs)
        kwparams_to_keep = {}
        for field_name, field in fields_dict(cls).items():
            if field.init and field_name in kwparams:
                kwparams_to_keep[field_name] = kwparams[field_name]
        return cls(**kwparams_to_keep)

    GECIClass = LightDependentGECI if light_dependent else GECI

    return init_from_kwparams(
        GECIClass,
        cal_model=init_from_kwparams(CalModel),
        exc_model=init_from_kwparams(ExcModel),
        bind_act_model=init_from_kwparams(BAModel),
        **kwparams,
    )


def _create_geci_fn(
    name,
    K_d,
    n_H,
    dFF_max,
    sigma_noise_rel,
    dFF_1AP_rel=None,
    ca_amp=None,
    t_on=None,
    t_off=None,
    extra_doc="",
):
    """convenience function for creating GECI model functions.

    K_d is in nM.

    dFF_1AP and sigma_noise are relative to GCaMP6s, since noise values obtained indirectly
    from Zhang et al., 2023, supp table 1, which is all relative to GCaMP6s.
    We take ΔF/F / SNR for 1 AP to get a measure of noise.

    We could have done something similar with supp table 1 from Dana 2019,
    but we assume measurements are more accurate from the later paper (as the paper explicitly states).
    Thus, we only use Dana 2019 for the absolute GCaMP6s measurement to scale
    the relative values given in Zhang et al., 2023 and for indiciators not in Zhang 2023
    supp table 1 (GCaMP3).
    """
    gcamp6s_dFF_1AP_dana2019 = 0.133
    gcamp6s_snr_1AP_dana2019 = 4.4
    sigma_gcamp6s = gcamp6s_dFF_1AP_dana2019 / gcamp6s_snr_1AP_dana2019
    sigma_noise = sigma_noise_rel * sigma_gcamp6s
    if dFF_1AP_rel is not None:
        dFF_1AP = dFF_1AP_rel * gcamp6s_dFF_1AP_dana2019
    else:
        dFF_1AP = None

    def geci_fn(
        light_dependent=False,
        doub_exp_conv=True,
        pre_existing_cal=False,
        K_d=K_d * nmolar,
        n_H=n_H,
        dFF_max=dFF_max,
        sigma_noise=sigma_noise,
        dFF_1AP=dFF_1AP,
        ca_amp=ca_amp,
        t_on=t_on,
        t_off=t_off,
        Ca_rest=50 * nmolar,
        kappa_S=110,
        gamma=292.3 / second,
        B_T=200 * umolar,
        dCa_T=7.6 * umolar,  # Lütke et al., 2013 and NAOMi code
        name=name,
        **kwparams,
    ) -> GECI:
        """Returns a (light-dependent) GECI model with specified submodel choices.
        Default parameters are taken from
        `NAOMi's code <https://bitbucket.org/adamshch/naomi_sim/src/25908cf432cd487fffe1f6442548d0fc8f4e8add/code/TimeTraceCode/calcium_dynamics.m?at=master#calcium_dynamics.m>`_
        (Song et al., 2021) as well as Dana et al., 2019 and Zhang et al., 2023.

        Only those parameters used in chosen model components apply.
        If the default is ``None``, then we don't have it fit yet.

        ``ca_amp``, ``t_on``, and ``t_off`` are given as in NAOMi, but are converted to
        the proper scale and units for the double exponential convolution model.
        Namely, ``A = ca_amp / (second / 100)`` and ``tau_[on|off] = second / t_[on|off]``.
        """
        # dt implicit in NAOMi's code, always s/100
        A = ca_amp / (second / 100) if ca_amp else None
        # had to reverse-engineer NAOMi code, which had surprising time constants
        tau_on = second / t_on if t_on else None
        tau_off = second / t_off if t_off else None

        return geci(
            light_dependent,
            doub_exp_conv,
            pre_existing_cal,
            K_d=K_d,
            n_H=n_H,
            dFF_max=dFF_max,
            sigma_noise=sigma_noise,
            dFF_1AP=dFF_1AP,
            A=A,
            tau_on=tau_on,
            tau_off=tau_off,
            Ca_rest=Ca_rest,
            kappa_S=kappa_S,
            gamma=gamma,
            B_T=B_T,
            dCa_T=dCa_T,
            name=name,
            **kwparams,
        )

    geci_fn.__doc__ += "\n\n" + (" " * 8) + extra_doc

    globals()[brian_safe_name(name.lower())] = geci_fn


# from NAOMi:
#                      K_d, n_H, dFF_max, sigma_noise_rel, dFF_1AP_rel, ca_amp, t_on, t_off
_create_geci_fn("GCaMP6f", 290, 2.7, 25.2, 1.24, 0.735, 76.1251, 0.8535, 98.6173)
_create_geci_fn("GCaMP6s", 147, 2.45, 27.2, 1, 1, 54.6943, 0.4526, 68.5461)
# noise, dFF for GCaMP3 from Dana 2019, since not in Zhang 2023
_create_geci_fn(
    "GCaMP3", 287, 2.52, 12, (3.9 / 2.1) / (13.3 / 4.4), 3.9 / 13.3, 0.05, 1, 1
)

# from NAOMi, but
# don't have double exponential convolution parameters for these:
_create_geci_fn(
    "OGB-1",
    250,
    1,
    14,
    1,
    extra_doc="*Don't know sigma_noise. Default is the GCaMP6s value.*",
)
_create_geci_fn(
    "GCaMP6-RS09",
    520,
    3.2,
    25,
    1,
    extra_doc="*Don't know sigma_noise. Default is the GCaMP6s value.*",
)
_create_geci_fn(
    "GCaMP6-RS06",
    320,
    3,
    15,
    1,
    extra_doc="*Don't know sigma_noise. Default is the GCaMP6s value.*",
)
_create_geci_fn("jGCaMP7f", 174, 2.3, 30.2, 0.72, 1.71)
_create_geci_fn("jGCaMP7s", 68, 2.49, 40.4, 0.33, 4.96)
_create_geci_fn("jGCaMP7b", 82, 3.06, 22.1, 0.25, 4.64)
_create_geci_fn("jGCaMP7c", 298, 2.44, 145.6, 0.39, 1.85)
