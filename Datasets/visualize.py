from matplotlib import pyplot as plt

def plot_input_to_well(X,Y,out_ending,well_names):
    cols=['CHK','PDC','PWH','PBH']
    for tag in well_names:
        i=1
        plt.figure()
        for col in cols:
            name_input=tag+'_'+col
            name_output=tag+'_'+out_ending
            plt.subplot(2,2,i)
            i+=1
            plt.scatter(X[name_input],Y[name_output],color='black')
            plt.title(name_input)
            plt.xlabel(name_input)
            plt.ylabel(name_output)
    #plt.show()
def plot_input_to_total(X,Y,out_ending,well_names):
    cols=['CHK','PDC','PWH','PBH']
    for tag in well_names:
        i=1
        plt.figure()
        for col in cols:
            name_input=tag+'_'+col
            name_output='GJOA'+'_'+out_ending
            plt.subplot(2,2,i)
            i+=1
            plt.scatter(X[name_input],Y[name_output],color='black')
            plt.title(name_input)
            plt.xlabel(name_input)
            plt.ylabel(name_output)
   # plt.show()