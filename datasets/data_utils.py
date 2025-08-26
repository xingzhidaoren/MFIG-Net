import numpy as np

def repeat_fewsamples(samples, label_ind=-1):
    '''
    Repeating the categories with few samples to overcome data imbalance
    '''

    classes = np.unique(samples[:,label_ind])

    cls_counts = []
    cls_inds = []
    for cls in classes:
        idx_cls = np.where(samples[:,label_ind]==cls)[0]
        cls_inds.append(idx_cls)
        cls_counts.append(len(idx_cls))

    max_count_ind = np.argmax(cls_counts)
    max_counts = cls_counts[max_count_ind]
    max_counts_inds = cls_inds[max_count_ind]
    data_repeats = []
    for k, cls in enumerate(classes):
        count = cls_counts[k]
        inds = cls_inds[k]
        if count == max_counts:
            continue
        
        # 计算需要重复的次数            
        repeat_times = max_counts // count     
        if repeat_times>0:      
            data_repeated = np.repeat(samples[inds], repeat_times, axis=0)
        else:
            data_repeated = samples[inds]

        remaining_samples = max_counts % count
        if remaining_samples > 0:
            additional_indices = np.random.choice(inds, remaining_samples, replace=False)
            data_repeated = np.vstack((data_repeated, samples[additional_indices]))

        data_repeats.append(data_repeated)

    data_repeats = np.vstack(data_repeats)
    balanced_data = np.vstack((data_repeats, samples[max_counts_inds]))
    return balanced_data