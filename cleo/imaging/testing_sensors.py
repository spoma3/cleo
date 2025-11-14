import sensors
from cleo.base import SynapseDevice
from cleo.light import LightDependent
import Dictionaries
from brian2 import nmolar, np, second, umolar

device = SynapseDevice()
s = sensors.Sensor(device, sigma_noise=1.0, dFF_1AP=0.2, location='cytoplasm')

model = input("GECI is: ")
# calcium_model = sensors.CalciumModel(model=model)
# ca_binding_activation_model = sensors.CalBindingActivationModel(model=model)
# exc_model = sensors.ExcitationModel()
cal_model = sensors.DynamicCalcium(Ca_rest=50 * nmolar, gamma=92.3/second, B_T=200*umolar, kappa_S=110, dCa_T=7.6*umolar)
# bind_model = sensors.DoubExpCalBindingActivation()
bind_model = sensors.NullBindingActivation()
sigma_noise = 1.0
dFF_1AP = 0.2
# g = sensors.GECI(s, cal_model=calcium_model, bind_act_model=ca_binding_activation_model, exc_model=exc_model,
#                  sigma_noise=sigma_noise, dFF_1AP=dFF_1AP, K_d=K_d, n_H=n_H, dFF_max=dFF_max)
x = float(input("brightness is: "))
w = int(input("wavelength is: "))
params_dict = Dictionaries.Light_Intensity_Dict[model]
params_w = Dictionaries.Wavelength_Dict[model]
# ldgeci = sensors.light_dependent_geci(name=model, doub_exp_conv=False, pre_existing_cal=False, bind_act_model=False,  Ca_rest=50 * nmolar,
#                          gamma=92.3/second, B_T=200*umolar, kappa_S=110, dCa_T=7.6 * umolar, x=x, w=w)

# sigma_noise=sigma_noise,dFF_1AP=dFF_1AP, K_d = params_dict['ec50'], n_H=params_dict['n'], dFF_max=params_dict['A'],
ldgeci = sensors.LightDependentGECI(name=model,
                          A=params_dict['A'], baseline=params_dict['baseline'], k=params_dict['k'], x=x, n=params_dict['n'],
                          ec50=params_dict['ec50'], baseline_w=params_w['baseline_w'], w=w, A1=params_w['A1'], sigma1=params_w['sigma1'],
                          A2=params_w['A2'], sigma2=params_w['sigma2'], mu1=params_w['mu1'], mu2=params_w['mu2'],
                          cal_model=cal_model, bind_act_model=bind_model, exc_model=sensors.NullExcitation())
val = ldgeci.fluorescence(model, x, w)
print(f"val is: {val}")