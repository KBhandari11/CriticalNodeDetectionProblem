import matplotlib.image as mpimg
import imageio.v2 as imageio
import torch
import numpy as np


length = np.array((len(mix),len(ba),len(finder)))
extent = (0, 25, 0, 25)
quickest = np.argmin(length)
images = []



BA = 337500
MIX =199000
attacker_BATypeGraph= torch.load('./model/DiffSize_BA_Graph/model_BAGraph_'+str(BA))#255000
attacker_differentGraph= torch.load('./model/DiffSize_Different_Graph/model_differentGraph_'+str(MIX))#104000(Motifs)

#file_list = ['ba_300_20_house_1', 'ba_300_40_house_2', 'ba_300_60_house_3', 'ba_300_80_house_4', 'ba_300_100_house_5', 'ba_300_20_grid_1', 'ba_300_40_grid_2', 'ba_300_60_grid_3', 'ba_300_80_grid_4', 'ba_300_100_grid_5', 'tree_8_20_cycle_1', 'tree_8_40_cycle_2', 'tree_8_60_cycle_3', 'tree_8_80_cycle_4', 'tree_8_100_cycle_5', 'tree_8_20_grid_1', 'tree_8_40_grid_2', 'tree_8_60_grid_3', 'tree_8_80_grid_4', 'tree_8_100_grid_5']
#['ba_60_10_house_1', 'ba_60_20_house_2', 'ba_60_30_house_3', 'ba_60_10_fan_1', 'ba_60_20_fan_2', 'ba_60_30_fan_3', 'ba_60_10_clique_1', 'ba_60_20_clique_2', 'ba_60_30_clique_3', 'ba_60_10_diamond_1', 'ba_60_20_diamond_2', 'ba_60_30_diamond_3', 'ba_60_10_cycle_1', 'ba_60_20_cycle_2', 'ba_60_30_cycle_3', 'ba_60_10_star_1', 'ba_60_20_star_2', 'ba_60_30_star_3', 'ba_60_10_grid_1', 'ba_60_20_grid_2', 'ba_60_30_grid_3']
file_list = ['ba_60_10_diamond_1']
file_path = "./Cross_Validation/GNNexplanation/New/"
x, mix,ba,finder,actions = visual_evaluation(file_path,file_list,attacker_differentGraph, attacker_BATypeGraph)
# Get x= x-axis, mix= Mixed_LCC, BA = BA_LCC, Finder = Finder_LCC, action = List of actions 
for i in range(length[quickest]-1):
    figure, axis = plt.subplots(1, 4, figsize=(20, 5),dpi=80, gridspec_kw={'width_ratios': [1,1,1, 1]})
    #figure.tight_layout()
    axis[0].imshow(mpimg.imread('./gif/mix/'+str(i)+'.png'), extent=extent)
    axis[0].set_title("Mix:"+str(actions[0][i]))

    axis[1].imshow(mpimg.imread('./gif/ba/'+str(i)+'.png'), extent=extent)
    axis[1].set_title("BA:"+str(actions[1][i]))

    axis[2].imshow(mpimg.imread('./gif/finder/'+str(i)+'.png'), extent=extent)
    axis[2].set_title("FINDER:" +str(actions[2][i]))
    
    
    axis[3].plot(x[:i],np.array(mix[:i]), 'green',label='Trained Attacker',marker='o')
    axis[3].plot(x[:i],np.array(ba[:i]), 'red',label='Trained Attacker [BA model]',marker='+')
    axis[3].plot(x[:i],np.array(finder[:i]), 'aquamarine',label='FINDER ReTrained',marker='x')
    axis[3].set_xlim(0, x[length[quickest]-1])
    axis[3].set_ylim(0, 1)
    axis[3].set_title("LCC vs Nodes")

    # Combine all the operations and display
    #plt.show()
    filename= "./gif/figure/"+str(i)
    plt.savefig(filename)
    images.append(imageio.imread(filename+".png"))
    plt.close()
kargs = { 'duration': 0.5 }
imageio.mimsave('~/Desktop/MAS/gif/diamond.gif', images, 'GIF', **kargs)