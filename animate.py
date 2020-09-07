import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# The source for LoopingPillowWriter was found on stack overflow but I 
# lost the original link and cannot properly attribute the author.

class LoopingPillowWriter(PillowWriter):
  def finish(self):
    self._frames[0].save(
      self._outfile, save_all=True, append_images=self._frames[1:],
      duration=int(1000 / self.fps), loop=0)
      
def animate_matrix_evolution(matrices,show=False,save=None,fps=5):
  
  fig, ax = plt.subplots()

  ax.axis('off')
  thing = ax.imshow(matrices[0],cmap="magma")

  def animate(i):
    thing.set_data(matrices[i])

  ani = FuncAnimation(fig, animate,frames=np.arange(len(matrices)))

  if save != None:
    ani.save(save+".gif",writer=LoopingPillowWriter(fps=fps))
  if show:
    plt.show()

def main():
  print("Test Animation Writer")

  adjacency_matrix = []

  for i in range(10):
    am = np.zeros((10,10))
    am[i] = 1
    adjacency_matrix.append(am)
    
  animate_matrix_evolution(adjacency_matrix,show=True,save="test")
  
if __name__ == "__main__":
  main()


