import igraph

"""
    please go to Python Packages and search for python-igraph.
    installation of cairocffi is also required.
    cairocffi also requires the installation of cairo by: brew install cairo
"""
import pandas as pd


def prediction_by_graph_theory(graph, nodes, no_of_predictions=5, epsilon=10 ** -5):
    valid_nodes = [node.upper() for node in nodes if node.upper() in graph.vs['name']]
    if len(valid_nodes) == 0:
        return "nil"

    neighbors = [graph.neighbors(node.upper()) for node in valid_nodes]
    g_sub = graph.subgraph([graph.vs[node]['name'] for node in set.union(*map(set, neighbors))] + valid_nodes)
    hashtags = [x['name'] for x in g_sub.vs if x['name'] not in valid_nodes]
    df = pd.DataFrame({'Hashtag': hashtags})
    for node in valid_nodes:
        df[node] = df['Hashtag'].apply(lambda x: g_sub.es[g_sub.get_eid(x, g_sub.vs.find(name=node))]['weight'] if g_sub.are_connected(x, g_sub.vs.find(name=node)) else 0)
        df[node] += epsilon
        df[node] = df[node] / df[node].sum()

    df['Hashtag'] = df['Hashtag'].apply(lambda x: x.strip())
    agg_dict = {}
    for node in valid_nodes:
        agg_dict[node] = ['sum']
    df = df.groupby(by='Hashtag').agg(agg_dict).reset_index()
    df.columns = ['Hashtag'] + valid_nodes
    df['Score'] = df.iloc[:, 1:].product(axis=1)
    df['norm_Score'] = df['Score'] / sum(df['Score'])
    df.sort_values(by=['Score'], ascending=False, inplace=True)
    df = df.head(no_of_predictions)
    return "\n".join(df['Hashtag'].values)


if __name__ == '__main__':
    G = igraph.Graph.Read_GML('hashtag_with_community.gml')
    print(prediction_by_graph_theory(G, ['amc', 'stock']))
