import os
import json
import subprocess
from datetime import datetime


def prepare_output_dir(args, user_specified_dir=None):
    """Prepare output directory.

    An output directory is created if it does not exist. Then the following
    infomation is saved into the directory:
      args.txt: command-line arguments
      git-status.txt: result of `git status`
      git-log.txt: result of `git log`
      git-diff.txt: result of `git diff`

    Args:
      args: dict that describes command-line arguments
      user_specified_dir: directory path
    """
    def timestamp():
        time = datetime.now()
        res = str(getattr(time, 'year'))
        for i in ['month', 'day', 'hour', 'minute']:
            res += '_'
            res += str(getattr(time, i))

        return res

    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError(
                    '{} is not a directory'.format(user_specified_dir))
        else:
            os.makedirs(user_specified_dir)
        outdir = user_specified_dir
    else:
        outdir = './result/' + timestamp()
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    # Save all the arguments
    with open(os.path.join(outdir, 'args.txt'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # Save `git status`
    with open(os.path.join(outdir, 'git-status.txt'), 'w') as f:
        f.write(subprocess.Popen('git status', stdout=subprocess.PIPE, shell=True).communicate()[0])

    # Save `git log`
    with open(os.path.join(outdir, 'git-log.txt'), 'w') as f:
        f.write(subprocess.Popen('git log', stdout=subprocess.PIPE, shell=True).communicate()[0])

    # Save `git diff`
    with open(os.path.join(outdir, 'git-diff.txt'), 'w') as f:
        f.write(subprocess.Popen('git diff', stdout=subprocess.PIPE, shell=True).communicate()[0])

    return outdir
