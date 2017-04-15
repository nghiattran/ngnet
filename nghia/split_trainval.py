import argparse
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('path', metavar='N', type=str,
                      help='Path to train.txt')
  parser.add_argument('-r', '--ratio', type=float, default=0.8, help='Ratio between val and train.')
  parser.add_argument('-v', '--verbose', action='store_true', help='Verbose.')
  args = parser.parse_args()

  if args.ratio < 0 or args.ratio > 1:
    print('Ratio must be between 0 and 1')
    exit(1)

  path = os.path.abspath(args.path)
  with open(path, 'r') as f:
    entries = f.readlines()
  split_pos = int(len(entries) * args.ratio)

  basedir = os.path.dirname(path)
  trainval_file= os.path.join(basedir, 'trainval.txt')
  with open(trainval_file, 'w') as f:
    f.writelines(entries)

  train_file = os.path.join(basedir, 'train.txt')
  with open(train_file, 'w') as f:
    f.writelines(entries[:split_pos])

  val_file = os.path.join(basedir, 'val.txt')
  with open(val_file, 'w') as f:
    f.writelines(entries[split_pos:])

  if args.verbose:
    if os.path.basename(path) != 'trainval.txt':
      print('Wrote %d entries to %s.' % (len(entries), trainval_file))
    print('Wrote %d (%.2f%%) entries to %s.' % (len(entries[:split_pos]), args.ratio * 100, train_file))
    print('Wrote %d (%.2f%%) entries to %s.' % (len(entries[split_pos:]), (1 - args.ratio) * 100, val_file))