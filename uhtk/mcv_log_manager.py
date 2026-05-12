import numpy as np
from uhtk.UTIL.colorful import *


def get_a_logger(logdir, color='k'):
    from uhtk.VISUALIZE.mcom import mcom
    mcv = mcom(
        path=f'{logdir}/mcom/',
        logdir=logdir,
        digit=16,
        rapid_flush=True,
        draw_mode='Img',
        resume_mod=False
    )
    mcv.rec_init(color=color)
    return mcv


class LogManager():
    def __init__(self, mcv, who=None, enable_smooth=False):
        if who is None:
            who = '*default*'
        self.who = who
        self.mcv = mcv
        self.enable_smooth = enable_smooth
        self.trivial_dict = {}
        self.smooth_trivial_dict = {}

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict:
                self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])

        print_buf = [f'[{self.who}] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, ' % (key, self.trivial_dict[key]))
            
            alpha = 0.98
            if key in self.smooth_trivial_dict:
                self.smooth_trivial_dict[key] = alpha*self.smooth_trivial_dict[key] + (1-alpha)*self.trivial_dict[key]
            else:
                self.smooth_trivial_dict[key] = self.trivial_dict[key]
            self.mcv.rec(self.trivial_dict[key], key)
            if self.enable_smooth:
                self.mcv.rec(self.smooth_trivial_dict[key], key + ' - smooth')
        if print:
            print紫(''.join(print_buf))

        self.mcv.rec_show()

        self.trivial_dict = {}
