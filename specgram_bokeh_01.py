from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftshift
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource

fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2) 
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time) 
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Sxx = signal.spectrogram(x, fs)
df_spectogram = pd.DataFrame(columns = ["Frequency","Time","Sxx"])

time = list(t)
freq = list(f)

df = pd.DataFrame(index = freq ,data = Sxx, columns = time)

plt.pcolormesh(t, f, Sxx)

df = pd.DataFrame(df.stack(), columns=['amp']).reset_index()

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

colors = ['#440154', '#404387', '#29788E', '#22A784', '#79D151', '#FDE724']
mapper = LinearColorMapper(palette=colors, low=df.amp.min(), high=df.amp.max())

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(title="Spectrogram",
           # x_range=time, y_range=freq,
			plot_width=1000, plot_height=600,
           tools=TOOLS, toolbar_location='below',
           tooltips=[('amp', '@amp%')])

p.circle(x="level_1", y="level_0", size=5,
       source=df,
       fill_color={'field': 'amp', 'transform': mapper},
       line_color=None)

show(p)
plt.show()