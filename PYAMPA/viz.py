import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as lines

import PYAMPA.utils as utils

def create_summary(pyampa_results_csv : str, output_directory : str):
    figfile = os.path.join(output_directory, 'summary_plots.png')

    df = pd.read_csv(pyampa_results_csv)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    df['Peptide Length'] = df['Peptide'].apply(len)
    plt.hist(df['Peptide Length'], bins=20, color='skyblue')
    plt.xlabel('Peptide Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Peptide Lengths')

    plt.subplot(2, 2, 2)
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    composition = df['Peptide'].apply(lambda x: [x.count(aa) for aa in amino_acids])
    composition = pd.DataFrame(composition.tolist(), columns=list(amino_acids))
    composition_sum = composition.sum()
    composition_sum.plot(kind='bar', color='lightcoral')
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.title('Amino Acid Composition of Predicted Peptides')

    plt.subplot(2, 2, 3)
    plt.hist(df['Average Torrent Index'], bins=20, color='lightgreen')
    plt.xlabel('Antimicrobial Index')
    plt.ylabel('Frequency')
    plt.title('Histogram of Antimicrobial Indices')

    plt.subplot(2, 2, 4)
    df['Hydrophobic Moment'] = df['Peptide'].apply(utils.calculate_hydrophobic_moment)
    plt.hist(df['Hydrophobic Moment'], bins=20, color='purple')
    plt.xlabel('Hydrophobic Moment')
    plt.ylabel('Frequency')
    plt.title('Histogram of Hydrophobic Moments')

    plt.tight_layout()
    plt.savefig(figfile, dpi=300)


def helical_wheel(sequence, colorcoding='rainbow', lineweights=True, filename=None, seq=False, moment=False):
    """A function to project a given peptide sequence onto a helical wheel plot. It can be useful to illustrate the
    properties of alpha-helices, like positioning of charged and hydrophobic residues along the sequence.

    :param sequence: {str} the peptide sequence for which the helical wheel should be drawn
    :param colorcoding: {str} the color coding to be used, available: *rainbow*, *charge*, *polar*, *simple*,
        *amphipathic*, *none*
    :param lineweights: {boolean} defines whether connection lines decrease in thickness along the sequence
    :param filename: {str} filename  where to safe the plot. *default = None* --> show the plot
    :param seq: {bool} whether the amino acid sequence should be plotted as a title
    :param moment: {bool} whether the Eisenberg hydrophobic moment should be calculated and plotted
    :return: a helical wheel projection plot of the given sequence (interactively or in **filename**)
    :Example:

    >>> helical_wheel('GLFDIVKKVVGALG')
    >>> helical_wheel('KLLKLLKKLLKLLK', colorcoding='charge')
    >>> helical_wheel('AKLWLKAGRGFGRG', colorcoding='none', lineweights=False)
    >>> helical_wheel('ACDEFGHIKLMNPQRSTVWY')

    .. image:: ../docs/static/wheel1.png
        :height: 300px
    .. image:: ../docs/static/wheel2.png
        :height: 300px
    .. image:: ../docs/static/wheel3.png
        :height: 300px
    .. image:: ../docs/static/wheel4.png
        :height: 300px

    .. versionadded:: v2.1.5
    """
    # color mappings
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    f_rainbow = ['#3e3e28', '#ffcc33', '#b30047', '#b30047', '#ffcc33', '#3e3e28', '#80d4ff', '#ffcc33', '#0047b3',
                 '#ffcc33', '#ffcc33', '#b366ff', '#29a329', '#b366ff', '#0047b3', '#ff66cc', '#ff66cc', '#ffcc33',
                 '#ffcc33', '#ffcc33']
    f_charge = ['#000000', '#000000', '#ff4d94', '#ff4d94', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
                '#000000', '#000000', '#000000', '#000000', '#000000', '#80d4ff', '#000000', '#000000', '#000000',
                '#000000', '#000000']
    f_polar = ['#000000', '#000000', '#80d4ff', '#80d4ff', '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff',
               '#000000', '#000000', '#80d4ff', '#000000', '#80d4ff', '#80d4ff', '#80d4ff', '#80d4ff', '#000000',
               '#000000', '#000000']
    f_simple = ['#ffcc33', '#ffcc33', '#0047b3', '#0047b3', '#ffcc33', '#7f7f7f', '#0047b3', '#ffcc33', '#0047b3',
                '#ffcc33', '#ffcc33', '#0047b3', '#ffcc33', '#0047b3', '#0047b3', '#0047b3', '#0047b3', '#ffcc33',
                '#ffcc33', '#ffcc33']
    f_none = ['#ffffff'] * 20
    f_amphi = ['#ffcc33', '#29a329', '#b30047', '#b30047', '#f79318', '#80d4ff', '#0047b3', '#ffcc33', '#0047b3',
               '#ffcc33', '#ffcc33', '#80d4ff', '#29a329', '#80d4ff', '#0047b3', '#80d4ff', '#80d4ff', '#ffcc33',
               '#f79318', '#f79318']
    t_rainbow = ['w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k', 'w', 'k', 'k', 'k', 'k', 'k']
    t_charge = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'w', 'w', 'w', 'k', 'w', 'w', 'w', 'w', 'w']
    t_polar = ['w', 'w', 'k', 'k', 'w', 'w', 'k', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w']
    t_simple = ['k', 'k', 'w', 'w', 'k', 'w', 'w', 'k', 'w', 'k', 'k', 'k', 'k', 'w', 'w', 'w', 'w', 'k', 'k', 'k']
    t_none = ['k'] * 20
    t_amphi = ['k', 'k', 'w', 'w', 'w', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'k', 'w', 'k', 'k', 'k', 'w', 'w']
    d_eisberg = utils.load_scale('eisenberg')  # eisenberg hydrophobicity values for HM
    
    if lineweights:
        lw = np.arange(0.1, 5.5, 5. / (len(sequence) - 1))  # line thickness array
        lw = lw[::-1]  # inverse order
    else:
        lw = [2.] * (len(sequence) - 1)
    
    # check which color coding to use
    if colorcoding == 'rainbow':
        df = dict(zip(aa, f_rainbow))
        dt = dict(zip(aa, t_rainbow))
    elif colorcoding == 'charge':
        df = dict(zip(aa, f_charge))
        dt = dict(zip(aa, t_charge))
    elif colorcoding == 'polar':
        df = dict(zip(aa, f_polar))
        dt = dict(zip(aa, t_polar))
    elif colorcoding == 'simple':
        df = dict(zip(aa, f_simple))
        dt = dict(zip(aa, t_simple))
    elif colorcoding == 'none':
        df = dict(zip(aa, f_none))
        dt = dict(zip(aa, t_none))
    elif colorcoding == 'amphipathic':
        df = dict(zip(aa, f_amphi))
        dt = dict(zip(aa, t_amphi))
    else:
        print("Unknown color coding, 'rainbow' used instead")
        df = dict(zip(aa, f_rainbow))
        dt = dict(zip(aa, t_rainbow))
    
    # degree to radian
    deg = np.arange(float(len(sequence))) * -100.
    deg = [d + 90. for d in deg]  # start at 270 degree in unit circle (on top)
    rad = np.radians(deg)
    
    # dict for coordinates and eisenberg values
    d_hydro = dict(zip(rad, [0.] * len(rad)))
    
    # create figure
    fig = plt.figure(frameon=False, figsize=(3, 3))
    ax = fig.add_subplot(111)
    old = None
    hm = list()
    
    # iterate over sequence
    for i, r in enumerate(rad):
        new = (np.cos(r), np.sin(r))  # new AA coordinates
        if i < 18:
            # plot the connecting lines
            if old is not None:
                line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                    linewidth=lw[i - 1])
                line.set_zorder(1)  # 1 = level behind circles
                ax.add_line(line)
        elif 17 < i < 36:
            line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                linewidth=lw[i - 1])
            line.set_zorder(1)  # 1 = level behind circles
            ax.add_line(line)
            new = (np.cos(r) * 1.2, np.sin(r) * 1.2)
        elif i == 36:
            line = lines.Line2D((old[0], new[0]), (old[1], new[1]), transform=ax.transData, color='k',
                                linewidth=lw[i - 1])
            line.set_zorder(1)  # 1 = level behind circles
            ax.add_line(line)
            new = (np.cos(r) * 1.4, np.sin(r) * 1.4)
        else:
            new = (np.cos(r) * 1.4, np.sin(r) * 1.4)
        
        # plot circles
        circ = patches.Circle(new, radius=0.1, transform=ax.transData, edgecolor='k', facecolor=df[sequence[i]])
        circ.set_zorder(2)  # level in front of lines
        ax.add_patch(circ)
        
        # check if N- or C-terminus and add subscript, then plot AA letter
        if i == 0:
            ax.text(new[0], new[1], sequence[i] + '$_N$', va='center', ha='center', transform=ax.transData,
                    size=12, color=dt[sequence[i]], fontweight='bold')
        elif i == len(sequence) - 1:
            ax.text(new[0], new[1], sequence[i] + '$_C$', va='center', ha='center', transform=ax.transData,
                    size=12, color=dt[sequence[i]], fontweight='bold')
        else:
            ax.text(new[0], new[1], sequence[i], va='center', ha='center', transform=ax.transData,
                    size=12, color=dt[sequence[i]], fontweight='bold')
        
        eb = d_eisberg[sequence[i]][0]  # eisenberg value for this AA
        hm.append([eb * new[0], eb * new[1]])  # save eisenberg hydrophobicity vector value to later calculate HM
        
        old = (np.cos(r), np.sin(r))  # save as previous coordinates
    
    # draw hydrophobic moment arrow if moment option
    if moment:
        v_hm = np.sum(np.array(hm), 0)
        x = .0333 * v_hm[0]
        y = .0333 * v_hm[1]
        ax.arrow(0., 0., x, y, head_width=0.04, head_length=0.03, transform=ax.transData,
                 color='k', linewidth=6.)
        desc = utils.calculate_hydrophobic_moment(sequence)
        if abs(x) < 0.2 and y > 0.:  # right positioning of HM text so arrow does not cover it
            z = -0.2
        else:
            z = 0.2
        plt.text(0., z, str(round(desc, 3)), fontdict={'fontsize': 20, 'fontweight': 'bold',
                                                                        'ha': 'center'})
    
    # plot shape
    if len(sequence) < 19:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    else:
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    
    if seq:
        plt.title(sequence, fontweight='bold', fontsize=20)
    
    # show or save plot
    if filename:
        plt.savefig(filename, dpi=150)
    else:
        pass