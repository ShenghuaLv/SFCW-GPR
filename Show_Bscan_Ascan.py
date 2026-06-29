import scipy.io as scio
import h5py
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import matplotlib.gridspec as gridspec
from Plot_Fig_Colors import colormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator


def Plot_Bscan2_Ascan(Bscan_dir1, Bscan_dir2):

    data1 = scio.loadmat(Bscan_dir1)
    Bscan_data1 = data1['data']
    Bscan_data1 = Bscan_data1[:3000, :]
    dt1 = data1['dt']

    data2 = scio.loadmat(Bscan_dir2)
    Bscan_data2 = data2['data2']
    Bscan_data2 = Bscan_data2[:750, :]
    print(Bscan_data1.shape)
    print(Bscan_data2.shape)


    colors, custom_cmap = colormap("color1")


    td = 15e-9

    # Bscan_data1 = BscanProcess(Bscan_data1, 1)
    # Bscan_data2 = BscanProcess(Bscan_data2, 1)
    Bscan_data1 = Bscan_data1 / np.max(Bscan_data1)
    Bscan_data2 = Bscan_data2 / np.max(Bscan_data2)

    Ascan1 = Bscan_data1[:, 128]
    Ascan2 = Bscan_data2[:, 128]


    fig = plt.figure(figsize=(10, 8), dpi=100, constrained_layout=True)

    gs = fig.add_gridspec(2, 4, width_ratios=[1, 0.05, 0.05, 0.6], wspace=0.05)



    # - Bscan1
    ax2 = fig.add_subplot(gs[0, 0])
    t1 = np.linspace(0, td, Bscan_data1.shape[0])
    im2 = ax2.imshow(Bscan_data1, aspect='auto', cmap='gray')
    data_height2, data_width2 = Bscan_data1.shape
    x_ticks2 = np.linspace(0, data_width2, 5)
    x_tick_labels2 = ["{:.0f}".format(xtick * 1) for xtick in x_ticks2]
    ax2.set_xticks(x_ticks2)
    ax2.set_xticklabels(x_tick_labels2, font={'family': 'Times New Roman', 'size': 16})
    y_ticks2 = np.linspace(0, data_height2, 6)
    y_tick_labels2 = ["{:.0f}".format(ytick * td * 1e9 / Bscan_data1.shape[0]) for ytick in y_ticks2]
    ax2.set_title('SFIP Bscan', fontdict={'family': 'Times New Roman', 'size': 16})
    ax2.set_yticks(y_ticks2)
    ax2.set_yticklabels(y_tick_labels2, font={'family': 'Times New Roman', 'size': 16})
    ax2.set_xlabel('Trace', font={'family': 'Times New Roman', 'size': 16})
    ax2.set_ylabel('Time (ns)', font={'family': 'Times New Roman', 'size': 16})
    ax2.tick_params(axis='both', labelsize=16)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_family('Times New Roman')

    # colorbar
    cbar_ax2 = fig.add_subplot(gs[0, 1])
    cbar2 = plt.colorbar(im2, cax=cbar_ax2, orientation='vertical', shrink=1.0, aspect=40, pad=0.01, format='%.1f')
    cbar2.set_label(label='Amplitude', font={'family': 'Times New Roman', 'size': 16})
    cbar2.ax.tick_params(labelsize=16)
    cbar2.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for label in cbar2.ax.get_yticklabels():
        label.set_family('Times New Roman')

    # A-scan
    ax5 = fig.add_subplot(gs[0, 3])
    ax5.plot(Ascan1, t1 * 1e9, color=colors[1], linewidth=1.5)
    ax5.invert_yaxis()
    ax5.set_ylim(t1[-1] * 1e9, t1[0] * 1e9)

    ax5.set_title('Ascan', fontsize=16, fontname='Times New Roman')
    ax5.set_xlabel('Amplitude', fontsize=16, fontname='Times New Roman')
    ax5.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax5.set_ylabel('Time (ns)', fontsize=16, fontname='Times New Roman')
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=5))


    ax5.tick_params(axis='both', labelsize=16)
    for label in ax5.get_xticklabels() + ax5.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax5.grid(True, alpha=0.3, linestyle='--')



    # - Bscan2
    ax3 = fig.add_subplot(gs[1, 0])
    t2 = np.linspace(0, td, Bscan_data2.shape[0])
    im3 = ax3.imshow(Bscan_data2, aspect='auto', cmap='gray')
    data_height3, data_width3 = Bscan_data2.shape
    x_ticks3 = np.linspace(0, data_width3, 5)
    x_tick_labels3 = ["{:.0f}".format(xtick * 1) for xtick in x_ticks3]
    ax3.set_xticks(x_ticks3)
    ax3.set_xticklabels(x_tick_labels3, font={'family': 'Times New Roman', 'size': 16})
    y_ticks3 = np.linspace(0, data_height3, 6)
    y_tick_labels3 = ["{:.0f}".format(ytick * td * 1e9 / Bscan_data2.shape[0]) for ytick in y_ticks3]
    ax3.set_title('SFCW Bscan', fontdict={'family': 'Times New Roman', 'size': 16})
    ax3.set_yticks(y_ticks3)
    ax3.set_yticklabels(y_tick_labels3, font={'family': 'Times New Roman', 'size': 16})
    ax3.set_xlabel('Trace', font={'family': 'Times New Roman', 'size': 16})
    ax3.set_ylabel('Time (ns)', font={'family': 'Times New Roman', 'size': 16})
    ax3.tick_params(axis='both', labelsize=16)
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_family('Times New Roman')

    # colorbar
    cbar_ax3 = fig.add_subplot(gs[1, 1])
    cbar3 = plt.colorbar(im3, cax=cbar_ax3, orientation='vertical', shrink=1.0, aspect=40, pad=0.01, format='%.1f')
    cbar3.set_label(label='Amplitude', font={'family': 'Times New Roman', 'size': 16})
    cbar3.ax.tick_params(labelsize=16)
    cbar3.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for label in cbar3.ax.get_yticklabels():
        label.set_family('Times New Roman')

    # A-scan
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.plot(Ascan2, t2 * 1e9, color=colors[6], linewidth=1.5)
    ax7.invert_yaxis()


    ax7.set_ylim(t1[-1] * 1e9, t1[0] * 1e9)

    ax7.set_title('Ascan', fontsize=16, fontname='Times New Roman')
    ax7.set_xlabel('Amplitude', fontsize=16, fontname='Times New Roman')
    ax7.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax7.set_ylabel('Time (ns)', fontsize=16, fontname='Times New Roman')
    ax7.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax7.tick_params(axis='both', labelsize=16)
    for label in ax7.get_xticklabels() + ax7.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax5.grid(True, alpha=0.3, linestyle='--')

    plt.show()




if __name__ == '__main__':

    filepath3 = './900Bscan_Single_900_syn.mat'
    filepath4 = './SFCWBscan_StartFreq_500.0StopFreq_3000.0Df_5.0_syn.mat'

    Plot_Bscan2_Ascan(filepath3, filepath4)