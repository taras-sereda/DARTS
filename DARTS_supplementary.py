__author__ = 'taras-sereda'

from collections import Counter
from scipy.io import loadmat
import numpy as np
import os
import scipy.stats

def read_gt(filename):
    with open(filename) as f:
        gt_map = dict([l.strip().split() for l in f.readlines()])
    ids = gt_map.keys()
    gt_weights = Counter(ids)

    return gt_map, ids, gt_weights


def get_all_probs(leaf_probs, tree):
    def get_prob(r, all_probs, tree):
        rnode = tree[r]

        num_children = rnode['num_children'][0][0][0]
        # print("current node {}, childrens {}".format(r, num_children))
        if num_children > 0:
            p = 0
            for i in range(num_children):
                # -1 for solving indexing problem natlab is 1_based Python 0_based
                c = rnode['children'][0][0][i] - 1
                all_probs = get_prob(c, all_probs, tree)
                p += all_probs[:, c]
            all_probs[:, r] = p

        return all_probs

    hs = tree['height']
    root = np.where(hs == np.max(hs))[0][0]
    n = leaf_probs.shape[0]
    m = tree.size

    all_probs = np.zeros((n, m))
    all_probs[:, :leaf_probs.shape[1]] = leaf_probs
    all_probs = get_prob(root, all_probs, tree)

    return all_probs


def info_rewards(tree):
    heights = tree['height']
    root_idx = np.where(heights == np.max(heights))[0][0]
    num_classes = tree.size

    is_leaf = heights == 0
    is_leaf = is_leaf.T
    num_leaf_descendants = get_all_probs(is_leaf, tree)
    num_leaves = np.nonzero(is_leaf)[0].size

    rewards = np.zeros((num_classes, 1))
    # the more descendants node has the less rewards it has.
    for i in range(num_classes):
        rewards[i] = np.log2(num_leaves * 1.0 / num_leaf_descendants[0][i])

    return rewards



def DARTS_predict(leaf_probs, rewards, tree):
    # Simply get the expected reward of each node for each image
    # and take max prediction value
    all_probs = get_all_probs(leaf_probs, tree)
    expected_values = all_probs * rewards.T[0]
    # preds in range [0:64]
    preds = np.argmax(expected_values, axis=1)

    return preds


def eval_reward(preds, labels, rewards, tree):


    num_examples = preds.size
    num_classes = tree.size
    # find how to flatten another way
    heights = np.array([l for i in tree['height'] for j in i for k in j for l in k]) + 1
    num_leaves = np.nonzero(heights == 1)[0].size

    pred_vec = np.zeros((num_examples,num_classes))
    pred_vec[np.arange(num_examples), preds] = 1

    gt_vec = np.zeros((num_examples, num_leaves))
    gt_vec[np.arange(num_examples), labels] = 1
    gt_vec_full = get_all_probs(gt_vec, tree)

    correct_vec = gt_vec_full * pred_vec
    acc = np.sum(correct_vec) / num_examples
    reward = np.sum(correct_vec * rewards.T) / num_examples

    height_pred_vec = pred_vec * heights
    height_counts = 1. * np.histogram(height_pred_vec, 1 + np.arange(np.max(heights) + 1))[0]
    height_correct_vec = correct_vec * heights
    height_goods = 1. * np.histogram(height_correct_vec, 1 + np.arange(np.max(heights) + 1))[0]
    height_acc = height_goods / height_counts
    height_portion = height_counts / num_examples

    return reward, acc, height_portion, height_acc

def binofit_scalar(x, n, alpha):
    '''Parameter estimates and confidence intervals for binomial data.
    (p,ci) = binofit(x,N,alpha)
    Source: Matlab's binofit.m
    Reference:
      [1]  Johnson, Norman L., Kotz, Samuel, & Kemp, Adrienne W.,
      "Univariate Discrete Distributions, Second Edition", Wiley
      1992 p. 124-130.
    http://books.google.com/books?id=JchiadWLnykC&printsec=frontcover&dq=Univariate+Discrete+Distributions#PPA131,M1
    Re-written by Santiago Jaramillo - 2008.10.06
    https://github.com/sjara/extracellpy/blob/master/extrastats.py
    '''

    if n < 1:
        Psuccess = np.nan
        ConfIntervals = (np.nan, np.nan)
    else:
        Psuccess = float(x)/n
        nu1 = 2 * x
        nu2 = 2 * (n - x + 1);
        F = scipy.stats.f.ppf(alpha / 2, nu1, nu2)
        lb = (nu1 * F) / (nu2 + nu1 * F)
        if x == 0:
            lb = 0
        nu1 = 2 * (x + 1)
        nu2 = 2 * (n - x)
        F = scipy.stats.f.ppf(1 - alpha / 2, nu1, nu2)
        ub = (nu1 * F) / (nu2 + nu1 * F)
        if x == n:
            ub = 1
        ConfIntervals = (lb, ub);
    return (Psuccess, ConfIntervals)


def DARTS_eval(leaf_probs, labels, tree, lambdas):
    # labels normalization to [0:64]
    labels -= 1
    tree_reward = info_rewards(tree)
    normed_rewards = tree_reward / max(tree_reward)
    num_heights = len(set([l for i in tree['height'] for j in i for k in j for l in k]))
    height_portions = np.zeros((lambdas.size, num_heights))
    height_accs = np.zeros((lambdas.size, num_heights))
    rewards = []
    accs = []
    for i,l in enumerate(lambdas):
        used_rewards = tree_reward + l
        preds = DARTS_predict(leaf_probs, used_rewards, tree)
        reward, acc, height_portion, height_acc = eval_reward(preds, labels, normed_rewards, tree)
        rewards.append(reward)
        accs.append(acc)

        height_portions[i,:] = height_portion
        height_accs[i,:] = height_acc

    return rewards, accs, height_portions, height_accs





def DARTS_bisection(accuracy_guaranty, leaf_probs, labels, tree
                    ,num_bs_iters, confidence):
    num_leaves = np.nonzero(tree['height'] == 0)[0].size
    num_examples = len(labels)
    rewards = info_rewards(tree)

    lambdas = []
    # labels normalization to [0:64]
    labels -= 1
    for k,i in enumerate(accuracy_guaranty):
        guarantee = i
        epsilon = 1 - guarantee

        desired_alpha = (1 - confidence) * 2
        min_lambda = 0
        max_lambda = ((1 - epsilon) * max(rewards) - min(rewards)) / epsilon

        for j in range(num_bs_iters):
            current_lambda = (min_lambda + max_lambda) * 1. / 2
            # Use transformed rewards when doing the binary search
            used_rewards = rewards + current_lambda
            # Make predictions and evaluate the reward.
            preds = DARTS_predict(leaf_probs, used_rewards, tree)
            reward, accuracy, _, _ = eval_reward(preds, labels, used_rewards, tree)
            _, acc_bounds = binofit_scalar(accuracy*num_examples, num_examples, desired_alpha)
            if acc_bounds[0] > guarantee:
                max_lambda = current_lambda
            else:
                min_lambda = current_lambda
        lambdas.append(max_lambda)
    return np.array(lambdas)


