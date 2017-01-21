''' Code to answer the question:
The Ramachandran plot visualizes the (theta,phi) angles of the amino acids from a data set of 
protein structures. Your task is to construct four 'variants' of the Ramachandran plot that
cover two residues, instead of one:
1. Plot the theta angles of residue i against the theta angles of residue i + 1.
2. Plot the theta angles of residue i against the phi angles of residue i + 1.
3. Plot the phi angles of residue i against the theta angles of residue i + 1.
4. Plot the phi angles of residue i against the phi angles of residue i + 1.
Here is some additional information and requirements regarding your task:
-- Implement everything in Python, and use Bio.PDB to parse the PDB files.
-- Analyse the plots. Can you distinguish regions that are highly or sparsely populated? Can you
explain what you see in terms of protein structure?
-- You are allowed to include additional plots and explore the data as you see fit.
For example, you could investigate the influence of 'special' amino acids such as Pro or Gly,
or the influence of solvent exposure.
-- Also, provide the four 'Ramachandran plots' for the PDB file 1ENH alone as well (the engrailed homeodomain) as test case.

This analysis was orginally used on the Top100 proteins data set (http://kinemage.biochem. duke.edu/databases/top100.php) or the larger Top8000 data set (http://kinemage.biochem.duke.edu/databases/top8000.php).

'''

import os
import time
import warnings
import pickle
import numpy
import Bio.PDB as PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt




# A future warning appears due to a problem with MatPlotLib.
# It doesn't affect the results, and MatPlotLib are currently working on fixing it. 
# Hence, the warning is hidden.
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# Prevents the peptide builder from outputting a list of atoms that required a name change.
warnings.simplefilter("ignore", PDBConstructionWarning)




####  This section of code calculates the angles and prepares them to be plotted ####


def dihedral_finder(model):
    '''
    For a given protein model, this function finds all the psi and phi angles.
    :param polypep:
    :return: phi_list; a list of the phi angles and residue names organised by polypeptide chain.
            psi_list; a list of the psi angles and residue names organised by polypeptide chain.
    '''

    # Create a list to be populated with phi & psi angles for each protein
    phi_list = []
    psi_list = []

    # Create a list of polypeptides in the model.
    ppb = PDB.PPBuilder()
    pp_list = ppb.build_peptides(model)

    phi = []
    psi = []

    # Iterate over polypeptide molecules if it contains more than 4 residues
    # (this is to ensure it is long enough to contribute to each of the 4 plots)
    for polypep in pp_list:
        if len(polypep) >= 4:
            l = len(polypep)
            # Iterate over each residue in the polypeptide.
            for residue in range(0, l):
                # 'Save' the atom vectors that will be used in both psi and phi calculations
                # (to avoid recalculation).
                try:
                    calpha = polypep[residue]['CA'].get_vector()
                    c = polypep[residue]['C'].get_vector()
                    n = polypep[residue]['N'].get_vector()
                    name = polypep[residue].get_resname()
                    # When residue = 0, polypep[residue-1] does not always return an index error.
                    # However, it will be pruned from the list of angles..
                    try:
                        # Calculate the phi angle using the C atom from the previous residue.
                        # If no atom exists, the except section will activate.
                        phi.append((PDB.calc_dihedral(polypep[residue - 1]['C'].get_vector(), n, calpha, c), name))
                    except (IndexError, KeyError):
                        # The atom information is missing.
                        phi.append(('M', name))
                    try:
                        # Calculate the psi angle in the same manner as for the phi angle.
                        # When residue = l,polypep[residue+1] returns an index error.
                        psi.append((PDB.calc_dihedral(n, calpha, c,
                                                      polypep[residue + 1]['N'].get_vector()),
                                    name))
                    except (IndexError, KeyError):
                        # The atom information is missing.
                        psi.append(('M', name))
                except KeyError:
                    # Necessary atom information is missing for both dihedral angle calculations. 
                    # So return 'NA' and skip to the next iteration of the for block
                    phi.append(('M', 'M'))
                    psi.append(('M', 'M'))

    # Here append is used as I want the whole list to be added to the end of phi_list as one entry.
    phi_list.append(phi)
    psi_list.append(psi)

    return [phi_list, psi_list]




def removeMissingAngles(tupleListX,tupleListY):
    '''
    TO BE USED IN A LATER FUNCTION
    :Param: Should be 1-dimensional lists tuples (phi angle OR psi angle OR 'M', residue name).
    Some angles are recorded as 'M'/Missing.  This is because the information needed from the atoms to calculate the
    angles is not present in the PDB file.  'M' cannot be plotted.  So we look in each list to be plotted (both x and y)
    and return the indices of any 'M' entries.  This entries are then removed, alongside the corresponding entry in the
    other list.
    '''
    # Create lists of only the angles
    x = [i[0] for i in tupleListX]
    y = [j[0] for j in tupleListY]
    # Generate a list of indices with a missing angle in either the x or y lists.
    Missing = [i for i, j in enumerate(x) if j == 'M']
    Missing.extend([i for i, j in enumerate(y) if j == 'M'])
    # Ensure each index appears once at most in the list, and remove indices from the end first (in order to maintain
    # consistent indexing in the section of the list that has not yet experienced deletions).
    for index in sorted(numpy.unique(Missing), reverse=True):
        del tupleListX[index]
        del tupleListY[index]



def getDihedralsList(folderpath):
    start_time = time.time()
    # Ignore construction warnings, as these refer only to the renaming of atoms.
    warnings.simplefilter("ignore", PDBConstructionWarning)
    # Create parser object
    parser = PDB.PDBParser()
    angleByChain = {'phi':[], 'psi':[]}
    proteinErrors = []
    # A counter to display the number of files processed.
    i = 0
    for file in os.listdir(folderpath):
        i += 1
        if i%500 == 0: print(i),
        try:
            # Get the structures contained
            struct = parser.get_structure('', folderpath+file)
            # dihedral_finder returns a list of the phi and psi dihedral angles, organised by chain.  As will not
            # organise my results via protein, I extend the list of angle rather than append.
            hold = dihedral_finder(struct[0])
            angleByChain['phi'].extend(hold[0])
            angleByChain['psi'].extend(hold[1])
        except (ValueError, TypeError):
            proteinErrors.append(file)
    # Return a list of problematic PDB files.
    if len(proteinErrors) > 0: print('The following proteins were not analysed due to issues with the PDB file: ', proteinErrors)

    # The angles stored in angleByChain are organised by chain.  This is to prevent (for example) the last phi angle
    # from one protein from being plotted against the first psi angle from the next protein (or chain).  As detailed in
    # my report, some beginning/end psi and/or phi points must be excluded.  The following code cycles through each
    # chain, selecting the appropriate values for plotting and adding them to the end of a list (which is no longer
    # organised by chain).
    # Each graph plots the relationship between different angles, and so the coordinates to be plotted are all different.
    # Thereofore, 4 sets of coordinates must be created.
    HH1X = []
    HH1Y = []
    for polyPep in range(len(angleByChain['phi'])):
        HH1X.extend(angleByChain['phi'][polyPep][1:-1])
        HH1Y.extend(angleByChain['phi'][polyPep][2:])

    HS1X = []
    HS1Y = []
    for polyPep in range(len(angleByChain['phi'])):
        HS1X.extend(angleByChain['phi'][polyPep][1:-2])
        HS1Y.extend(angleByChain['psi'][polyPep][2:-1])

    SH1X = []
    SH1Y = []
    for polyPep in range(len(angleByChain['phi'])):
        SH1X.extend(angleByChain['psi'][polyPep][:-1])
        SH1Y.extend(angleByChain['phi'][polyPep][1:])

    SS1X = []
    SS1Y = []
    for polyPep in range(len(angleByChain['psi'])):
        SS1X.extend(angleByChain['psi'][polyPep][0:-2])
        SS1Y.extend(angleByChain['psi'][polyPep][1:-1])

    # The same process is repeated for the original Ramachandran Plot.
    RCDX = []
    RCDY = []
    for polyPep in range(len(angleByChain['phi'])):
        RCDX.extend(angleByChain['phi'][polyPep][1:-1])
        RCDY.extend(angleByChain['psi'][polyPep][1:-1])

    # Remove any coordinates that contain missing angles.
    removeMissingAngles(HH1X, HH1Y)
    removeMissingAngles(HS1X, HS1Y)
    removeMissingAngles(SH1X, SH1Y)
    removeMissingAngles(SS1X, SS1Y)
    removeMissingAngles(RCDX, RCDY)

    # Return the complete list of angles to plotted, alongside their residue names.  This is designed to allow the list
    # to filtered based on residue name without recalculating the entire list of angles.  This is important,
    # particularly for the 8000 protein data set.
    print("--- %s seconds ---" % (time.time() - start_time))
    return HH1X, HH1Y, HS1X, HS1Y, SH1X, SH1Y, SS1X, SS1Y, RCDX, RCDY






#### This part of the code creates the functions that plot the graphs. ####

#   There is a separate function for the testCase data, as this does not require the LogNorm colour bar scaling.
#   There is also a separate function to plot residue specific graphs, as the data must first be filtered before
#   plotting.



def removeNonRes(tupleListX,tupleListY, residue):
    '''
    :Param: Should be 1-dimensional lists tuples (phi angle OR psi angle OR 'M', residue name).


    Used to filter the list to retrieve coordinates that contain at least one angle referring to the given residue.
    '''
    # Create lists of only the residue names
    x = [i[1] for i in tupleListX]
    y = [j[1] for j in tupleListY]
    # Generate a list of indices with a Gly residue in either the x or y lists.
    getResList = [i for i, j in enumerate(x) if j == residue]
    getResList.extend([i for i, j in enumerate(y) if j == residue])
    # Ensure each index appears once at most in the list, and remove indices from the end first (in order to maintain
    # consistent indexing in the section of the list that has not yet experienced deletions).

    xRes = []
    yRes = []
    for index in sorted(numpy.unique(getResList)):
        xRes.append(tupleListX[index])
        yRes.append(tupleListY[index])

    return xRes, yRes



def plotBasedOnRes(picklefilename, Res):
    '''
    :picklefilename: name of the pickle file, located in the same folder as the py script, without the '.p' suffix.
    :Res: Name of a residue, as abbreviated in PDB files, in all CAPS.  For example, 'GLY'

    This section produces hist2D plots with LogNorm from the ----picklefilename.p---- data analysed in the
    ProteinDihedral section after filtering for a given residue.

    Note:  vmax is set to 15800 for the lognorm plots so that all may be compared to the top8000 data set via the same
    colour bar

    This function is primarly intended for the top8000 data.
    '''

    pi = numpy.pi

    fullHH1X, fullHH1Y, fullHS1X, fullHS1Y, fullSH1X, fullSH1Y, fullSS1X, fullSS1Y, fullRCDX, fullRCDY= pickle.load(open(picklefilename+'.p', "rb"))



    HH1X, HH1Y = removeNonRes(fullHH1X, fullHH1Y, Res)
    HS1X, HS1Y = removeNonRes(fullHS1X, fullHS1Y, Res)
    SH1X, SH1Y = removeNonRes(fullSH1X, fullSH1Y, Res)
    SS1X, SS1Y = removeNonRes(fullSS1X, fullSS1Y, Res)
    RCDX, RCDY = removeNonRes(fullRCDX, fullRCDY, Res)


    # start_time = time.time()
    binsize = 175



    # plot with row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(9,9))
    f.canvas.set_window_title(picklefilename+'Variants'+Res)

    HH = ax1.hist2d(x=[i[0] for i in HH1X], y=[j[0] for j in HH1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax1.set_title("Variant 1: $\phi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)


    HS = ax2.hist2d(x=[i[0] for i in HS1X], y=[j[0] for j in HS1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax2.set_title("Variant 2: $\phi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    SH = ax3.hist2d(x=[i[0] for i in SH1X], y=[j[0] for j in SH1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax3.set_title("Variant 3: $\psi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    SS = ax4.hist2d(x=[i[0] for i in SS1X], y=[j[0] for j in SS1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax4.set_title("Variant 4: $\psi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    ax1.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax1.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax4.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax4.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax1.set_xlim([-pi,pi])
    ax1.set_ylim([-pi,pi])
    ax4.set_xlim([-pi,pi])
    ax4.set_ylim([-pi,pi])
    ax1.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', pad=10, labelsize=14)
    ax4.tick_params(axis='x', pad=10, labelsize=14)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, hspace=.2)
    cbar_axis = f.add_axes([0.1, 0.03, .8, 0.035])
    f.colorbar(HH[3], cax=cbar_axis, orientation='horizontal')
    #f.colorbar(im[3], cax=ax1)
    #f.tight_layout(h_pad=2)

    plt.figure(picklefilename+'RCD'+Res)
    plt.xlabel('$\phi_i$', fontsize=20)
    plt.ylabel('$\psi_{i}$', fontsize=16).set_rotation(0)
    RCD = plt.hist2d(x=[i[0] for i in RCDX], y=[j[0] for j in RCDY], bins=binsize, norm=LogNorm(vmax=15800))
    plt.xticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.yticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.tick_params(axis='both', which='major', labelsize=16, pad=12)
    plt.tight_layout(h_pad=1)
    plt.colorbar(RCD[3])



    plt.show()





def plotLogHist2d(picklefilename):
    '''
    :picklefilename: name of the pickle file, located in the same folder as the py script, without the '.p' suffix.

    This section produces the plots from the ----picklefilename.p---- data analysed in the ProteinDihedral section.

    Note:  vmax is set to 15800 for the lognorm plots so that all may be compared to the top8000 data set via the same
    colour bar

    This function is primarly intended for the top8000 data.
    '''
    pi = numpy.pi

    HH1X, HH1Y, HS1X, HS1Y, SH1X, SH1Y, SS1X, SS1Y, RCDX, RCDY = pickle.load(open(picklefilename+".p", "rb"))

    binsize = 175


    # plot with row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(9,9))
    f.canvas.set_window_title(picklefilename+'Variants')

    HH = ax1.hist2d(x=[i[0] for i in HH1X], y=[j[0] for j in HH1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax1.set_title("Variant 1: $\phi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)


    HS = ax2.hist2d(x=[i[0] for i in HS1X], y=[j[0] for j in HS1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax2.set_title("Variant 2: $\phi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    SH = ax3.hist2d(x=[i[0] for i in SH1X], y=[j[0] for j in SH1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax3.set_title("Variant 3: $\psi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    SS = ax4.hist2d(x=[i[0] for i in SS1X], y=[j[0] for j in SS1Y], bins=binsize, norm=LogNorm(vmax=15800))
    ax4.set_title("Variant 4: $\psi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)

    ax1.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax1.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax4.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax4.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax1.set_xlim([-pi,pi])
    ax1.set_ylim([-pi,pi])
    ax4.set_xlim([-pi,pi])
    ax4.set_ylim([-pi,pi])
    ax1.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', pad=10, labelsize=14)
    ax4.tick_params(axis='x', pad=10, labelsize=14)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, hspace=.2)
    cbar_axis = f.add_axes([0.1, 0.03, .8, 0.035])
    f.colorbar(HH[3], cax=cbar_axis, orientation='horizontal')

    plt.figure(picklefilename+'RCD')
    plt.xlabel('$\phi_i$', fontsize=20)
    plt.ylabel('$\psi_{i}$', fontsize=16).set_rotation(0)
    RCD = plt.hist2d(x=[i[0] for i in RCDX], y=[j[0] for j in RCDY], bins=binsize, norm=LogNorm(vmax=15800))
    plt.xticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.yticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.tick_params(axis='both', which='major', labelsize=16, pad=12)
    plt.tight_layout(h_pad=1)
    plt.colorbar(RCD[3])



    plt.show()





def plotNormalHist2d(picklefilename):
    '''
    :picklefilename: name of the pickle file, located in the same folder as the py script, without the '.p' suffix.

    This section produces the plots from the ----picklefilename.p---- data analysed in the ProteinDihedral section.

    It is primarily intended for the testCase data.
    '''
    pi = numpy.pi

    HH1X, HH1Y, HS1X, HS1Y, SH1X, SH1Y, SS1X, SS1Y, RCDX, RCDY= pickle.load(open(picklefilename+".p", "rb"))


    binsize = 25

    # plot with row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(9,9))
    f.canvas.set_window_title(picklefilename+'Variants')
    HH = ax1.hist2d(x=[i[0] for i in HH1X], y=[j[0] for j in HH1Y], bins=binsize, cmin=1)
    ax1.set_title("Variant 1: $\phi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)
    HH[3].set_clim(vmin=0, vmax=15)

    HS = ax2.hist2d(x=[i[0] for i in HS1X], y=[j[0] for j in HS1Y], bins=binsize, cmin=1)
    ax2.set_title("Variant 2: $\phi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)
    HS[3].set_clim(vmin=0, vmax=15)

    SH = ax3.hist2d(x=[i[0] for i in SH1X], y=[j[0] for j in SH1Y], bins=binsize, cmin=1)
    ax3.set_title("Variant 3: $\psi_i$ against $\phi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)
    SH[3].set_clim(vmin=0, vmax=15)

    SS = ax4.hist2d(x=[i[0] for i in SS1X], y=[j[0] for j in SS1Y], bins=binsize, cmin=1)
    ax4.set_title("Variant 4: $\psi_i$ against $\psi_{i+1}$", loc='left', fontsize=12, linespacing=0.5)
    SS[3].set_clim(vmin=0, vmax=15)

    ax1.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax1.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_yticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax3.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax3.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax4.set_xticks([-pi, -pi/2, 0, pi/2, pi], minor=False)
    ax4.set_xticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'], minor=False)
    ax1.set_xlim([-pi,pi])
    ax1.set_ylim([-pi,pi])
    ax4.set_xlim([-pi,pi])
    ax4.set_ylim([-pi,pi])
    ax1.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    ax3.tick_params(axis='x', pad=10, labelsize=14)
    ax4.tick_params(axis='x', pad=10, labelsize=14)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, hspace=.2)
    cbar_axis = f.add_axes([0.1, 0.03, .8, 0.035])
    f.colorbar(HH[3], cax=cbar_axis, orientation='horizontal')


    plt.figure(picklefilename+'RCD')
    plt.xlabel('$\phi_i$', fontsize=20)
    plt.ylabel('$\psi_{i}$', fontsize=16).set_rotation(0)
    RCD = plt.hist2d(x=[i[0] for i in RCDX], y=[j[0] for j in RCDY], bins=binsize, cmin=1)
    plt.xticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.yticks([-pi, -pi/2, 0, pi/2, pi],
               [r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    plt.tick_params(axis='both', which='major', labelsize=16, pad=12)
    plt.tight_layout(h_pad=1)
    RCD[3].set_clim(vmin=0, vmax=15)
    plt.colorbar(RCD[3])

    plt.show()








####  This part of the code runs the functions defined above, in order to actually produce the data and create the plots. ####

# The data created is stored in pickle files so that the lengthy getDihedralsList function need not be called every time a graph is plotted.

#testCaseDict = getDihedralsList('testCase/')
#pickle.dump(testCaseDict, open("testCaseResult.p", "wb"))


#top100dict = getDihedralsList('top100H/')
#pickle.dump(top100dict, open("top100Result.p", "wb"))

#top8000dict = getDihedralsList('top8000/')
#pickle.dump(top8000dict, open("top8000Result.p", "wb"))

plotNormalHist2d('testCaseResult')
plotBasedOnRes('top8000Result', 'GLY')
plotBasedOnRes('top8000Result', 'PRO')
plotLogHist2d('top8000Result')