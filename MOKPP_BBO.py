import sys
import operator
import random
import math
import numpy as np
import graph_tool.all as gt

import matplotlib.pyplot as plt
import pandas as pd
import colorsys

import timeit
from collections import Counter

def deep_getsizeof(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([deep_getsizeof(v, seen) for v in obj.values()])
        size += sum([deep_getsizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += deep_getsizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([deep_getsizeof(i, seen) for i in obj])
    return size


def get_distance_sum(net, v_set):
    return sum([sum(gt.shortest_distance(net, source=v, target=v_set, directed=net.is_directed())) for v in v_set])


def get_scores(net, objfunc):
    return {
        'Degree': lambda net:
            [(v.out_degree() * (1.0 if net.is_directed() else 2)) / (net.num_vertices() * net.num_vertices()) for v in net.vertices()],
        'Closeness': lambda net:
            gt.closeness(net),
        'Betweenness': lambda net:
            gt.betweenness(net)[0],
        'PageRank': lambda net:
            gt.pagerank(net)
    }[objfunc](net)


def get_score(net, v, objfunc):
    if "Degree" == objfunc:
        return (v.out_degree() * (1.0 if net.is_directed() else 2)) / (net.num_vertices() * net.num_vertices())
    elif "Closeness" == objfunc:
        return gt.closeness(net)[v]
    elif "Betweenness" == objfunc:
        return gt.betweenness(net)[0][v]
    elif "PageRank" == objfunc:
        return gt.pagerank(net)[v]

    #score = {
    #    'Degree': lambda net:
    #    (v.out_degree() * (1.0 if net.is_directed() else 2)) / (net.num_vertices() * net.num_vertices()),
    #    'Closeness': lambda net:
    #        gt.closeness(net)[v],
    #    'Betweenness': lambda net:
    #        gt.betweenness(net)[v],
    #    'PageRank': lambda net:
    #        gt.pagerank(net)[v]
    #}[objective_func](net)

    return None


def dominance_checker(p, q):
    p_dominated = False
    q_dominated = False
    for i in range(len(p)):
        if p[i] < q[i]:
            p_dominated = True
        elif p[i] > q[i]:
            q_dominated = True

        if p_dominated and q_dominated:
            return 0
    if p_dominated:
        if not q_dominated:
            return 1
    elif q_dominated:
        return -1
    return 0


def non_dominated_rank(objscores):
    rank = [None for p in objscores]
    fronts = [[]]
    sp = [[] for p in objscores]
    np = [0 for p in objscores]
    for p in range(len(objscores)):
        for q in range(len(objscores)):
            if (p != q):
                domflag = dominance_checker(objscores[p], objscores[q])
                if domflag == -1: # q dominates p
                    sp[p].append(q)
                elif domflag == 1: # p dominates q
                    np[p] += 1
        if np[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while True:
        queue = []
        for p in fronts[i]:
            for q in sp[p]:
                np[q] -= 1
                if np[q] == 0:
                    rank[q] = i + 1
                    queue.append(q)
        if len(queue) > 0:
            i += 1
            fronts.append(queue)
        else:
            break
    return fronts, rank


def build_key(net, v_set):
    v_labels = net.vertex_properties[net.vertex_properties.keys()[0]]
    return '-'.join([v_labels[i] for i in sorted(v_set)])


def roulette_selection(weights, sorted=True):
    sorted_indexed_weights = list(enumerate(weights))
    if not sorted:
        sorted_indexed_weights = sorted(sorted_indexed_weights, key=operator.itemgetter(1))
    weight_sum = 0.0
    for i, w in sorted_indexed_weights:
        weight_sum += w

    value = random.random() * weight_sum

    for i, w in sorted_indexed_weights:
        value -= w
        if (value <= 0):
            return i
    return len(weights) - 1


def initialize_population(pop_size, sol_size, actors_with_prob_selection):
    weights = [i[1] for i in actors_with_prob_selection]

    #(frequency) rw_test = {i: 0 for i in range(len(weights))}
    #(frequency) print(rw_test)
    population = []
    for s in range(pop_size):
        solution = []
        for d in range(sol_size):
            i = roulette_selection(weights)
            #(frequency) rw_test[i] += 1
            while True:
                s = actors_with_prob_selection[i][0]
                if s in solution:
                    i = (i if i != 0 else len(actors_with_prob_selection)) - 1
                else:
                    solution.append(s)
                    break
        solution = sorted(solution)
        population.append(solution)
    #(frequency) print(rw_test)
    #(frequency) plt.bar(range(len(rw_test)), rw_test.values(), align='center')
    #(frequency) plt.xticks(range(len(rw_test)), rw_test.keys())

    #(frequency) plt.show()

    return population


def make_supernet(net, v_set):
    # pos = gt.sfdp_layout(net)
    # gt.graph_draw(net, pos, vertex_text=net.vertex_properties["name"], output="graph-draw-sfdp.png")
    supernet = net.copy()
    sv_set = [int(vi) for vi in v_set]
    supernode = max({vi: supernet.vertex(vi).out_degree() for vi in sv_set}.items(), key=operator.itemgetter(1))[0]
    super_key = build_key(supernet, sv_set)
    supernet.vertex_properties[supernet.vertex_properties.keys()[0]][supernode] = super_key
    is_adjust_require = True if (supernode + len(v_set) >= supernet.num_vertices()) else False
    rm_sv_set = [v for v in reversed(sorted(list(set(sv_set) - set([supernode]))))]
    for vi in rm_sv_set:
        vi_neighbours_to_process = list(set([int(u) for u in supernet.vertex(vi).out_neighbours()]) - set(sv_set))
        for u in vi_neighbours_to_process:
            if supernet.edge(supernet.vertex(supernode), supernet.vertex(u)) is None:
                supernet.add_edge(supernode, u, False)
    for vi in rm_sv_set:
        supernet.remove_vertex(vi, fast=True)

    if is_adjust_require:
        v_props = supernet.vertex_properties[supernet.vertex_properties.keys()[0]]
        for i in range(supernet.num_vertices() - 1, -1, -1):
            if v_props[i] == super_key:
                supernode = supernet.vertex(i)
                break

    # UNNECESSARY CODE - START ################################################
    # pos = gt.sfdp_layout(supernet)
    # gt.graph_draw(supernet, pos, vertex_text=supernet.vertex_properties["name"], output="graph-draw-sfdp-supernet.png")

    # pos = gt.sfdp_layout(net)
    # gt.graph_draw(net, pos, vertex_text=net.vertex_properties["name"], output="graph-draw-sfdp-net.png")
    # UNNECESSARY CODE - END ##################################################

    # print({l:supernet.vertex_properties[supernet.vertex_properties.keys()[0]][l] for l in range(supernet.num_vertices())})
    # print("106 ", supernet.vertex_properties[supernet.vertex_properties.keys()[0]][supernode], " ", supernode)

    return supernet, supernode


def find_key_set(net, objfuncs, num_key_actors, frac_serviceables):
    print(objfuncs)
    effobjfuncs = [objfunc[0] for objfunc in objfuncs if objfunc[0] != "Distance"]
    t_start = timeit.default_timer()
    num_serviceables = math.ceil(net.num_vertices() * frac_serviceables)
    # ,Closeness:Max
    objscores = [get_scores(net, objfunc) for objfunc in effobjfuncs]
    objscores_mat = []
    for v in range(net.num_vertices()):
        objscores_mat.append([objscores[i][v] for i in range(len(objscores))])
    # print(objscores_mat)
    fronts, rank = non_dominated_rank(objscores_mat)
    # print(fronts)
 
    # fronts_selection_prob = [(i, (1.0 - (i / fronts_size))) for i in range(fronts_size)]
    # print(fronts_selection_prob)

    # print(rank)
    actors_with_prob_selection = [(i, (1.0 - (rank[i] / (max(rank) + 1)))) for i in range(len(rank))]
    # print(actors_with_prob_selection)

    ranked_actors_with_prob_selection = sorted(actors_with_prob_selection, key=lambda kv: kv[1], reverse=True)
    servicable_actors_with_prob_selection = ranked_actors_with_prob_selection[0: num_serviceables]
    # print(servicable_actors_with_prob_selection)

    # '''Initialize population'''
    pop_size = 30
    # print(num_key_actors)
    pop_current = initialize_population(pop_size, num_key_actors, servicable_actors_with_prob_selection)
    pop_current = [[build_key(net, sol), sol, None] for sol in pop_current]

    # num_elites = 2
    max_gen = 5000 #00000
    gen_similarity_gap = 2000
    # best_cost = None
    # last_best_gen = 0
    current_gen = 0
    # theoretical_objective_best_value = 1.0

    colours = ["#000000", "#020000", "#040000", "#060000", "#080000", "#0A0000", "#0C0000", "#0E0000", "#100000",
               "#120000", "#140000", "#160000", "#180000", "#1A0000", "#1C0000", "#1E0000", "#200000", "#220000",
               "#240000", "#260000", "#280000", "#2A0000", "#2C0000", "#2E0000", "#300000", "#320000", "#340000",
               "#360000", "#380000", "#3A0000", "#3C0000", "#3E0000", "#400000", "#410000", "#440000", "#460000",
               "#480000", "#490000", "#4C0000", "#4E0000", "#500000", "#510000", "#540000", "#560000", "#580000",
               "#590000", "#5C0000", "#5E0000", "#600000", "#610000", "#640000", "#660000", "#680000", "#690000",
               "#6C0000", "#6E0000", "#700000", "#710000", "#740000", "#760000", "#780000", "#790000", "#7C0000",
               "#7E0000", "#800000", "#820000", "#830000", "#860000", "#880000", "#8A0000", "#8C0000", "#8E0000",
               "#900000", "#920000", "#930000", "#960000", "#980000", "#9A0000", "#9C0000", "#9E0000", "#A00000",
               "#A20000", "#A30000", "#A60000", "#A80000", "#AA0000", "#AC0000", "#AE0000", "#B00000", "#B20000",
               "#B30000", "#B60000", "#B80000", "#BA0000", "#BC0000", "#BE0000", "#C00000", "#C20000", "#C30000",
               "#C60000", "#C80000", "#CA0000", "#CC0000", "#CE0000", "#D00000", "#D20000", "#D30000", "#D60000",
               "#D80000", "#DA0000", "#DC0000", "#DE0000", "#E00000", "#E20000", "#E30000", "#E60000", "#E80000",
               "#EA0000", "#EC0000", "#EE0000", "#F00000", "#F20000", "#F30000", "#F60000", "#F80000", "#FA0000",
               "#FC0000", "#FE0000", "#FF0000", "#FF0200", "#FF0400", "#FF0600", "#FF0800", "#FF0B00", "#FF0D00",
               "#FF0F00", "#FF1000", "#FF1200", "#FF1400", "#FF1600", "#FF1900", "#FF1B00", "#FF1D00", "#FF1F00",
               "#FF2000", "#FF2200", "#FF2400", "#FF2600", "#FF2800", "#FF2B00", "#FF2D00", "#FF2F00", "#FF3000",
               "#FF3200", "#FF3400", "#FF3600", "#FF3900", "#FF3B00", "#FF3D00", "#FF3F00", "#FF4100", "#FF4200",
               "#FF4400", "#FF4600", "#FF4800", "#FF4B00", "#FF4D00", "#FF4F00", "#FF5100", "#FF5200", "#FF5400",
               "#FF5600", "#FF5900", "#FF5B00", "#FF5D00", "#FF5F00", "#FF6100", "#FF6200", "#FF6400", "#FF6600",
               "#FF6800", "#FF6B00", "#FF6D00", "#FF6F00", "#FF7100", "#FF7200", "#FF7400", "#FF7600", "#FF7900",
               "#FF7B00", "#FF7D00", "#FF7F00", "#FF8100", "#FF8300", "#FF8400", "#FF8600", "#FF8800", "#FF8B00",
               "#FF8D00", "#FF8F00", "#FF9100", "#FF9300", "#FF9400", "#FF9600", "#FF9900", "#FF9B00", "#FF9D00",
               "#FF9F00", "#FFA100", "#FFA300", "#FFA400", "#FFA600", "#FFA800", "#FFAB00", "#FFAD00", "#FFAF00",
               "#FFB100", "#FFB300", "#FFB400", "#FFB600", "#FFB900", "#FFBB00", "#FFBD00", "#FFBF00", "#FFC100",
               "#FFC300", "#FFC400", "#FFC600", "#FFC800", "#FFCB00", "#FFCD00", "#FFCF00", "#FFD100", "#FFD300",
               "#FFD400", "#FFD600", "#FFD900", "#FFDB00", "#FFDD00", "#FFDF00", "#FFE100", "#FFE300", "#FFE400",
               "#FFE600", "#FFE800", "#FFEB00", "#FFED00", "#FFEF00", "#FFF100", "#FFF300", "#FFF400", "#FFF600",
               "#FFF900", "#FFFB00", "#FFFD00", "#FFFF00"]

    weights = [i[1] for i in servicable_actors_with_prob_selection]
    # prob_immigration = [None for i in range(pop_size)]
    # prob_emigration = [None for i in range(pop_size)]
    sol_fitness_prob = [None for i in range(pop_size)]
    prob_mutation = 0.15
    sol_cache = {}
    gen_fronts_best = []
    gen_new = {}
    while current_gen < max_gen:
        sys.stdout.write("\033[F") #back to previous line
        sys.stdout.write("\033[K") #clear line
        print("GEN: {}".format(current_gen + 1))
        c = 0
        for sol_tuple in pop_current: # sol_tuple: index [ 0: Solution_key, 1: vertex indexes, 2: solution objective cost
            sol_tuple[2] = sol_cache.get(sol_tuple[0], None)
            if sol_tuple[2] is None:
                c = c + 1
                supernet, supernode = make_supernet(net, sol_tuple[1])
                sol_tuple[2] = [get_score(supernet, supernode, objfunc[0]) if objfunc[0] != "Distance" else (get_distance_sum(net, sol_tuple[1]) / num_key_actors) for objfunc in objfuncs]
                sol_cache[sol_tuple[0]] = sol_tuple[2]

        gen_new[current_gen] = c
        fronts, rank = non_dominated_rank([objscore[2] for objscore in pop_current])
        for fi in fronts[0]:
            gen_fronts_best.append((current_gen + 1, pop_current[fi][0], pop_current[fi][2]))
        fronts_size = len(fronts)

        # Plot #######################################################
        df = []
        for f in range(fronts_size):
            for fi in fronts[f]:
                y = [(f+1)] + pop_current[fi][2]
                df.append(y)
        df = pd.DataFrame(df, columns=["fronts", "PageRank", "Distance"])
        df = df.sort_values(["PageRank", "Distance"], ascending=False)

        groups = df.groupby('fronts')

        fig, ax = plt.subplots()
        # fig.tight_layout()

        # fig = plt.figure()
        fig.set_size_inches(5.5, 3)
        # ax.set_color_cycle(colors)
        ax.margins(0.05)
        for name, group in groups:
            line_style = '-' if name == 1 else '-'
            line_width = 2 if name == 1 else 0.35
            alpha = 1 if name == 1 else 0.75 #(0.65 - 0.25 * (name / fronts_size))
            cidx = name * math.floor(len(colours) / (1.15 * fronts_size))
            color = colours[cidx]
            ax.plot(group.PageRank, group.Distance, marker='x', linestyle=line_style, linewidth=line_width, ms=8, label=name, alpha=alpha, color=color)
            # ax.set_xlim(0, 0.5)
            # ax.set_ylim(0, 120)

        ax.grid(True)
        ax.set_xlabel("Cumulative PageRank for the key actors")
        ax.set_ylabel("Average distance between key actors")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.925, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Front")

        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle(':')

        # ax.set_xlim([0, 1.0])

        fig.savefig("plots/mo/PR-dist-" + str(current_gen).zfill(4) + ".pdf", dpi=100, bbox_inches='tight', transparent="True", pad_inches=0.01)
        plt.close(fig)
        # plt.show()
        # Plot - end #################################################

        # print(fronts)
        if fronts_size > 1:
            for f in range(fronts_size):
                for fi in fronts[f]:
                    sol_fitness_prob[fi] = 1.0 - f / (fronts_size - 1)
        else:
            sol_fitness_prob[fi] = random.random()
            print(340, " ", current_gen)

        # print(303, prob_emigration)
        # print(304, prob_immigration)

        pop_nextgen = pop_current.copy()
        current_gen_sol_cache = {}
        for s in fronts[0]:
            current_gen_sol_cache[pop_nextgen[s][0]] = True
        for dom_front in fronts[1:]:
            for s in dom_front:
                for d in range(num_key_actors):
                    if prob_mutation >= random.random():
                        i = roulette_selection(weights)
                        k = actors_with_prob_selection[i][0]
                        while k in pop_nextgen[s][1][:d]:
                            i = (i if i != 0 else len(actors_with_prob_selection)) - 1
                            k = actors_with_prob_selection[i][0]
                        pop_nextgen[s][1][d] = k
                    elif sol_fitness_prob[s] > random.random():
                        # Migration
                        i = roulette_selection(sol_fitness_prob)
                        k = pop_current[i][1][d]
                        dn = d
                        while k in pop_nextgen[s][1][:d]:
                            dn = (dn if dn != 0 else num_key_actors) - 1
                            k = pop_current[i][1][dn]
                            if dn == d:
                                i = roulette_selection(sol_fitness_prob)
                                dn = d + 1
                        pop_nextgen[s][1][d] = k
                    else:
                        # No operation check for duplicate
                        k = pop_nextgen[s][1][d]
                        if k in pop_nextgen[s][1][:d]:
                            i = roulette_selection(weights)
                            k = actors_with_prob_selection[i][0]
                            while k in pop_nextgen[s][1][:d]:
                               i = (i if i != 0 else len(actors_with_prob_selection)) - 1
                               k = actors_with_prob_selection[i][0]
                            pop_nextgen[s][1][d] = k

                pop_nextgen[s][1] = sorted(pop_nextgen[s][1])
                ct = Counter(pop_nextgen[s][1])
                if (len(ct) != num_key_actors):
                   return
                pop_nextgen[s][0] = build_key(net, pop_nextgen[s][1])
                
                # Reduce duplicate solution
                while pop_nextgen[s][0] in current_gen_sol_cache:
                    d = random.randint(0, num_key_actors - 1)
                    i = roulette_selection(weights)
                    k = actors_with_prob_selection[i][0]
                    while k in pop_nextgen[s][1]:
                        i = (i if i != 0 else len(actors_with_prob_selection)) - 1
                        k = actors_with_prob_selection[i][0]
                    pop_nextgen[s][1][d] = k
                    pop_nextgen[s][1] = sorted(pop_nextgen[s][1])
                    pop_nextgen[s][0] = build_key(net, pop_nextgen[s][1])
                current_gen_sol_cache[pop_nextgen[s][0]] = True

        pop_current = pop_nextgen

        current_gen += 1

    # print(gen_new)
    final_gen = []
    for gbst in gen_fronts_best:
        if gbst[0] == current_gen:
            final_gen.append(gbst)
    print(final_gen)

    # Plot #######################################################
    df = [[gbst[0]] + gbst[2] for gbst in gen_fronts_best]
    df = pd.DataFrame(df, columns=["Generation", "PageRank", "Distance"])
    df = df.sort_values(["Generation", "PageRank", "Distance"], ascending=True)
    df.to_csv("gen_front_best.csv", sep=',', encoding='utf-8')

    groups = df.groupby('Generation')

    fig, ax = plt.subplots()
    # fig.tight_layout()

    fig.set_size_inches(5.5, 3.85)
    # ax.set_color_cycle(colors)
    ax.margins(0.05)
    best_gen = max([s[0] for s in gen_fronts_best])
    for name, group in groups:
        line_style = '-' # if name == 1 else '-'
        line_width = 2 if name == best_gen else 0.35
        alpha = 1 if name == best_gen else 0.75  # (0.65 - 0.25 * (name / fronts_size))
        cidx = best_gen - name + 1 # * math.floor(len(colours) / (max(gen_fronts_best[0])))
        cidx = math.floor(cidx / (math.ceil(best_gen + 1) / len(colours)))
        # print(cidx, " ", best_gen, " ", name, " ", len(colours))
        color = colours[cidx]
        ax.plot(group.PageRank, group.Distance, marker='x', linestyle=line_style, linewidth=line_width, ms=8,
                label=name, alpha=alpha, color=color)

    ax.grid(True)
    ax.set_xlabel("Cumulative PageRank for the key actors")
    ax.set_ylabel("Average distance between key actors")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.925, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Front")

    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle(':')

    # ax.set_xlim([0, 1.0])

    fig.savefig("plots/mo/Gen-PR-dist.pdf", dpi=100, bbox_inches='tight', transparent="True", pad_inches=0.01)
    plt.close(fig)
    # plt.show()
    # Plot - end #################################################

    # return pop_current[0][1], pop_current[0][2], last_best_gen


def main():
    file_path = sys.argv[1]
    network_type = sys.argv[2]
    objfuncs = sys.argv[3]
    num_key_actors = int(sys.argv[4])
    frac_serviceables = float(sys.argv[5])

    objfuncs = [objfunc.split(":") for objfunc in objfuncs.split(",")]

    # net = gt.load_graph_from_csv(file_name=file_path, directed=(True if network_type == "Directed" else False))
    net = gt.load_graph(file_name=file_path)

    # xx = net.vertex_properties[net.vertex_properties.keys()[0]]
    # print({v:xx[v] for v in range(net.num_vertices())})

    # pr = gt.pagerank(net)
    # print(pr.a)
    # pr.a = 1000 * pr.a
    # v_cols = sns.heatmap(pr.a, vmin=0, vmax=1)
    # # prscores.a = prscores.a * 100
    # pos = gt.sfdp_layout(net) # fruchterman_reingold_layout
    # # gt.graph_draw(net, pos, vertex_size=prscores, vertex_fill_color=prscores, vertex_text=net.vertex_properties[net.vertex_properties.keys()[0]], output="dolphins-draw-sfdp-net.pdf")
    # gt.graph_draw(net, output_size=(1000, 1000), vertex_size=pr, vertex_fill_color=pr, vcmap=plt.cm.gist_heat_r, output="plots/soc-wiki-vote.pdf")
    # gt.graph_draw(net, pos, vertex_size=pr, vertex_fill_color=pr, vcmap=plt.cm.gist_heat_r, output="plots/wikipedia_human_navigation_paths.pdf")

    t_start = timeit.default_timer() # time.time()
    # actors, optimum_val, best_gen = \
    find_key_set(net, objfuncs, num_key_actors, frac_serviceables)
    t_end = timeit.default_timer() # time.time()
    print(t_end - t_start)

    # # Stat code start ############################################
    # scores = {
    #     'Degree': lambda net:
    #         net.out_degrees(),
    #     'Closeness': lambda net:
    #         gt.closeness(net),
    #     'Betweenness': lambda net:
    #         gt.betweenness(net)[0],
    #     'PageRank': lambda net:
    #         gt.pagerank(net)
    # }[objective_func](net)
    # scores = {v: scores[v] for v in net.vertices()}
    # scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    # num_verts = len(scores)
    # ord_verts = [scores[i][0] for i in range(num_verts)]
    #
    # last_idx = 0
    # for v in actors:
    #     if ord_verts.index(v) > last_idx:
    #         last_idx = ord_verts.index(v)
    #
    # # Stat code end ###############################################
    #
    # with open("sim-result.txt", "a") as file:
    #     file.write("\"{}\",\"{}\",{},{},\"{}\",{},{},{},{},{}\n".format(file_path, objfuncs, num_key_actors, frac_serviceables, build_key(net, actors), optimum_val, best_gen, (t_end - t_start), num_verts, last_idx))


main()
# http://networkrepository.com
# "datasets/networks/prison.gml" "Undirected" "PageRank,Distance" 5 1.0
