import os
import copy
import pandas as pd

opt = dict()

opt['dataset'] = '../data/cora' 
opt['hidden_dim'] = 16
opt['input_dropout'] = 0.5
opt['dropout'] = 0
opt['optimizer'] = 'rmsprop'
opt['lr'] = 0.05
opt['decay'] = 5e-4
opt['self_link_weight'] = 1.0
opt['pre_epoch'] = 100
opt['epoch'] = 100
opt['iter'] = 1
opt['use_gold'] = 1
opt['draw'] = 'smp'
opt['tau'] = 0.1
opt['log'] = 'cora.txt'

if os.path.exists(opt['log']):
    os.system('rm %s' %opt['log'])

def generate_command(opt):
    cmd = 'python3 train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

for k in range(100):
    seed = k + 1
    opt['seed'] = seed
    run(opt)

# save log message
df = pd.read_table(opt['log'], names=['base', 'gnnp', 'gnnq'], sep='\t')
base, gnnp, gnnq = df.base.values, df.gnnp.values, df.gnnq.values
line1 = 'base max_acc:{:.2f}, avg_acc:{:.2f}, std:{:.2f}'.format(base.max(), base.mean(), base.std())
line2 = 'gnnp max_acc:{:.2f}, avg_acc:{:.2f}, std:{:.2f}'.format(gnnp.max(), gnnp.mean(), gnnp.std())
line3 = 'gnnq max_acc:{:.2f}, avg_acc:{:.2f}, std:{:.2f}'.format(gnnq.max(), gnnq.mean(), gnnq.std())
print(line1 +'\n' + line2 + '\n' + line3)
df = df.append(pd.DataFrame([[line1,None,None], [line2,None,None], [line3,None, None]], columns=['base', 'gnnp', 'gnnq']))
df.to_csv(opt['log'], index=None)