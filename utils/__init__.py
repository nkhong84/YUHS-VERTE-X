
import pickle
import gzip

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def save_pickle(fileapth,obj):
  # save and compress.
  with gzip.open(fileapth, 'wb') as f:
    pickle.dump(obj, f)

def read_pickle(fileapth):
  with gzip.open(fileapth,"rb") as f:
    data = pickle.load(f)

  return data