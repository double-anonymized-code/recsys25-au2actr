import numpy as np

from au2actr.constants import SESSION_LEN, NEGSAM_POP, NEGSAM_UNIFORM


def train_sample(uid, nxt_idx, data, seqlen, n_items, user_tracks,
                 **kwargs):
    track_ids_map = data['track_ids_map']
    user_sessions = data['user_sessions'][uid]
    user_ids_map = data['user_ids_map']
    bll_weights = data['track_bll_weights']['train']
    spread_weights = data['track_spread_weights']['train']
    pos_spread_weights = data['track_pos_spread_weights']['train']
    num_favs = kwargs['num_favs']
    internal_track_ids = np.arange(1, n_items + 1)
    negsam_strategy = kwargs['negsam_strategy']
    track_pops = None
    if negsam_strategy == NEGSAM_POP:
        track_pops = kwargs['norm_track_popularities']
    seq_in = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    seq_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_pos = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    pos_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_neg = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    nxt = user_sessions[nxt_idx]
    idx = seqlen - 1
    for sess in reversed(user_sessions[:nxt_idx]):
        seq_in_track_ids = [track_ids_map[tid] for tid in sess['track_ids']]
        seq_in[idx][:len(seq_in_track_ids)] = seq_in_track_ids
        seq_actr_bll[idx][:len(seq_in_track_ids)] = \
            [bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx]
             else 0. for tid in sess['track_ids']]
        seq_pos[idx][:len(nxt['track_ids'])] = [track_ids_map[tid]
                                                for tid in nxt['track_ids']]
        pos_actr_bll[idx][:len(nxt['track_ids'])] = \
            [bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx] \
                 else 0. for tid in nxt['track_ids']]
        # nxt_set = set([track_ids_map[tid] for tid in nxt['track_ids']])
        if negsam_strategy == NEGSAM_UNIFORM:
            neg_ids = np.random.choice(internal_track_ids, size=SESSION_LEN)
            for j, neg in enumerate(neg_ids):
                # while neg in nxt_set:
                while neg in user_tracks:
                    neg_ids[j] = neg = np.random.choice(internal_track_ids)
        else:
            neg_ids = np.random.choice(internal_track_ids, size=SESSION_LEN,
                                       p=track_pops)
            for j, neg in enumerate(neg_ids):
                while neg in user_tracks:
                    neg_ids[j] = neg = np.random.choice(internal_track_ids,
                                                        p=track_pops)
        seq_neg[idx] = neg_ids
        nxt = sess
        idx -= 1
        if idx == -1:
            break
    seq_actr_spread = spread_weights[uid][nxt_idx]
    pos_actr_spread = np.concatenate(
        (spread_weights[uid][nxt_idx][1:], pos_spread_weights[uid][nxt_idx]),
        axis=0)
    # sort personal items by descending BLL
    sorted_entries = sorted(bll_weights[uid][nxt_idx].items(),
                            key=lambda x: x[1], reverse=True)[:num_favs]
    if len(sorted_entries) < num_favs:
        topbll_ids = np.zeros(num_favs, dtype=np.int32)
        topbll_scores = np.zeros(num_favs, dtype=np.float32)
        topbll_ids[:len(sorted_entries)] = [track_ids_map[tid]
                                            for tid, _ in sorted_entries]
        topbll_scores[:len(sorted_entries)] = [score for _, score in sorted_entries]
    else:
        topbll_ids = np.array([track_ids_map[tid] for tid, _ in sorted_entries])
        topbll_scores = np.array([score for _, score in sorted_entries])
    out = (seq_in, seq_actr_bll, seq_actr_spread,
           pos_actr_bll, pos_actr_spread,
           topbll_ids, topbll_scores, user_ids_map[uid], seq_pos, seq_neg)
    return out


def valid_sample(uid, data, seqlen, nxt_idx, n_items, user_tracks, **kwargs):
    track_ids_map = data['track_ids_map']
    user_sessions = data['user_sessions'][uid]
    user_ids_map = data['user_ids_map']
    mode = kwargs['mode']
    bll_weights = data['track_bll_weights'][mode]
    spread_weights = data['track_spread_weights']['test']
    pos_spread_weights = data['track_pos_spread_weights']['test']
    num_favs = kwargs['num_favs']
    internal_track_ids = np.arange(1, n_items + 1)
    negsam_strategy = kwargs['negsam_strategy']
    track_pops = None
    if negsam_strategy == NEGSAM_POP:
        track_pops = kwargs['norm_track_popularities']
    seq_in = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    seq_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_pos = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    pos_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    seq_neg = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    nxt = user_sessions[nxt_idx]
    idx = seqlen - 1
    for sess in reversed(user_sessions[:nxt_idx]):
        seq_in_track_ids = [track_ids_map[tid] for tid in sess['track_ids']]
        seq_in[idx][:len(seq_in_track_ids)] = seq_in_track_ids
        seq_actr_bll[idx][:len(seq_in_track_ids)] = [
            bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx] \
                else 0.0 for tid in sess['track_ids']]
        seq_pos[idx][:len(nxt['track_ids'])] = [track_ids_map[tid]
                                                for tid in nxt['track_ids']]
        pos_actr_bll[idx][:len(nxt['track_ids'])] = \
            [bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx] \
                 else 0. for tid in nxt['track_ids']]
        nxt_set = set([track_ids_map[tid] for tid in nxt['track_ids']])
        if negsam_strategy == NEGSAM_UNIFORM:
            neg_ids = np.random.choice(internal_track_ids, size=SESSION_LEN)
            for j, neg in enumerate(neg_ids):
                while neg in nxt_set:
                # while neg in user_tracks:
                    neg_ids[j] = neg = np.random.choice(internal_track_ids)
        else:
            neg_ids = np.random.choice(internal_track_ids, size=SESSION_LEN,
                                       p=track_pops)
            for j, neg in enumerate(neg_ids):
                # while neg in user_tracks:
                while neg in nxt_set:
                    neg_ids[j] = neg = np.random.choice(internal_track_ids,
                                                        p=track_pops)
        seq_neg[idx] = neg_ids
        nxt = sess
        idx -= 1
        if idx == -1:
            break
    seq_actr_spread = spread_weights[uid][nxt_idx]
    pos_actr_spread = np.concatenate(
        (spread_weights[uid][nxt_idx][1:], pos_spread_weights[uid][nxt_idx]),
        axis=0)

    # sort personal items by descending BLL
    sorted_entries = sorted(bll_weights[uid][nxt_idx].items(),
                            key=lambda x: x[1], reverse=True)[:num_favs]
    if len(sorted_entries) < num_favs:
        topbll_ids = np.zeros(num_favs, dtype=np.int32)
        topbll_scores = np.zeros(num_favs, dtype=np.float32)
        topbll_ids[:len(sorted_entries)] = [track_ids_map[tid] for tid, _ in
                                            sorted_entries]
        topbll_scores[:len(sorted_entries)] = [score for _, score in
                                               sorted_entries]
    else:
        topbll_ids = np.array([track_ids_map[tid] for tid, _ in sorted_entries])
        topbll_scores = np.array([score for _, score in sorted_entries])
    out = (seq_in, seq_actr_bll, seq_actr_spread,
           pos_actr_bll, pos_actr_spread,
           topbll_ids, topbll_scores, user_ids_map[uid], seq_pos, seq_neg)
    return out


def test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    track_ids_map = data['track_ids_map']
    user_sessions = data['user_sessions'][uid]
    user_ids_map = data['user_ids_map']
    mode = kwargs['mode']
    bll_weights = data['track_bll_weights'][mode]
    spread_weights = data['track_spread_weights']['test']
    num_favs = kwargs['num_favs']
    seq_in = np.zeros(shape=[seqlen, SESSION_LEN], dtype=np.int32)
    seq_actr_bll = np.zeros(shape=[seqlen, SESSION_LEN],
                            dtype=np.float32)
    idx = seqlen - 1
    for sess in reversed(user_sessions[:nxt_idx]):
        seq_in_track_ids = [track_ids_map[tid] for tid in sess['track_ids']]
        seq_in[idx][:len(seq_in_track_ids)] = seq_in_track_ids
        seq_actr_bll[idx][:len(seq_in_track_ids)] = [
            bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx] \
                else 0.0 for tid in sess['track_ids']]
        idx -= 1
        if idx == -1:
            break
    seq_actr_spread = spread_weights[uid][nxt_idx]
    # sort personal items by descending BLL
    sorted_entries = sorted(bll_weights[uid][nxt_idx].items(),
                            key=lambda x: x[1], reverse=True)[:num_favs]
    if len(sorted_entries) < num_favs:
        topbll_ids = np.zeros(num_favs, dtype=np.int32)
        topbll_scores = np.zeros(num_favs, dtype=np.float32)
        topbll_ids[:len(sorted_entries)] = [track_ids_map[tid] for tid, _ in
                                            sorted_entries]
        topbll_scores[:len(sorted_entries)] = [score for _, score in
                                               sorted_entries]
    else:
        topbll_ids = np.array([track_ids_map[tid] for tid, _ in sorted_entries])
        topbll_scores = np.array([score for _, score in sorted_entries])
    out = (seq_in, seq_actr_bll, seq_actr_spread, topbll_ids, topbll_scores,
           user_ids_map[uid], uid)
    return out
