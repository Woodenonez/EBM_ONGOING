

def plot_on_sdd(axes, img, label, cell_idx, traj, e_grid, prob_map):
    ax1, ax2, ax3 = axes
    ax1.imshow(img, cmap='gray')
    ax1.plot(traj[-1,0], traj[-1,1], 'ko', label='current')
    ax1.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax1.plot(label[0], label[1], 'bo', label="ground truth")
    ax1.legend()
    ax1.legend(prop={'size': 14}, loc='upper right')
    ax1.set_aspect('equal', 'box')

    ax2.imshow(e_grid, cmap='gray')
    ax2.plot(cell_idx[0,0], cell_idx[0,1], 'rx')

    ax3.imshow(prob_map, cmap='hot')
    ax3.plot(cell_idx[0,0], cell_idx[0,1], 'gx')

    ax1.set_title('Real world')
    ax2.set_title('Energy grid')
    ax3.set_title('Probability map')