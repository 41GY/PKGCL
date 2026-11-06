import argparse
import csv
import statistics
from collections import deque
from pathlib import Path


CANCER = "TEST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_gmt(path):
    pathways = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"GMT line {line_number} has fewer than three fields")
            pathways.append({"name": fields[0], "genes": {gene.strip() for gene in fields[2:] if gene.strip()}})
    names = [pathway["name"] for pathway in pathways]
    if len(names) != 50 or len(names) != len(set(names)):
        raise ValueError(f"Expected 50 unique Hallmark pathways, found {len(names)} rows and {len(set(names))} unique names")
    return pathways


def jaccard_matrix(pathways):
    size = len(pathways)
    matrix, intersections = [[0.0] * size for _ in range(size)], [[0] * size for _ in range(size)]
    for source in range(size):
        matrix[source][source] = 1.0
        intersections[source][source] = len(pathways[source]["genes"])
        for target in range(source + 1, size):
            overlap = len(pathways[source]["genes"] & pathways[target]["genes"])
            union = len(pathways[source]["genes"] | pathways[target]["genes"])
            matrix[source][target] = matrix[target][source] = overlap / union if union else 0.0
            intersections[source][target] = intersections[target][source] = overlap
    return matrix, intersections


def topk_edges(matrix, k):
    edges = set()
    for source in range(len(matrix)):
        for target in sorted((target for target in range(len(matrix)) if target != source), key=lambda target: (-matrix[source][target], target))[:k]:
            edges.add(tuple(sorted((source, target))))
    return edges


def graph_stats(node_count, edges, matrix):
    adjacency = [set() for _ in range(node_count)]
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    degrees, visited, component_sizes = [len(neighbors) for neighbors in adjacency], set(), []
    for start in range(node_count):
        if start not in visited:
            queue, size = deque([start]), 0
            visited.add(start)
            while queue:
                node = queue.popleft()
                size += 1
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            component_sizes.append(size)
    weights = [matrix[source][target] for source, target in edges]
    return {
        "node_count": node_count,
        "undirected_edge_count": len(edges),
        "directed_edge_count_without_self_loops": 2 * len(edges),
        "density": 2 * len(edges) / (node_count * (node_count - 1)),
        "degree_min": min(degrees),
        "degree_median": statistics.median(degrees),
        "degree_mean": statistics.fmean(degrees),
        "degree_max": max(degrees),
        "isolated_node_count": sum(degree == 0 for degree in degrees),
        "connected_component_count": len(component_sizes),
        "largest_component_size": max(component_sizes),
        "edge_weight_min": min(weights) if weights else None,
        "edge_weight_median": statistics.median(weights) if weights else None,
        "edge_weight_mean": statistics.fmean(weights) if weights else None,
        "edge_weight_max": max(weights) if weights else None,
        "degrees": degrees,
    }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a Hallmark pathway Jaccard graph.")
    parser.add_argument("--cancer", default=CANCER, help="Write the shared graph into one cancer output directory")
    parser.add_argument("--gmt", type=Path, default=PROJECT_ROOT / "data" / "hallmark.gmt")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or PROJECT_ROOT / "phase4" / "output" / args.cancer
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pathways = read_gmt(args.gmt)
    names = [pathway["name"] for pathway in pathways]
    matrix, intersections = jaccard_matrix(pathways)
    with (args.output_dir / "pathway_jaccard_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pathway", *names])
        writer.writerows([name, *[format(value, ".17g") for value in row]] for name, row in zip(names, matrix))

    candidate_edges = {k: topk_edges(matrix, k) for k in range(1, 11)}
    candidate_rows = [
        {"method": "topk_symmetric_union", "parameter": k, **{key: value for key, value in graph_stats(50, edges, matrix).items() if key != "degrees"}}
        for k, edges in candidate_edges.items()
    ]
    write_csv(args.output_dir / "graph_candidate_qc.csv", list(candidate_rows[0]), candidate_rows)

    connected = [row for row in candidate_rows if row["isolated_node_count"] == 0 and row["connected_component_count"] == 1]
    if not connected:
        raise RuntimeError("No connected Top-K candidate found in K=1..10")
    selected = min(connected, key=lambda row: (int(row["parameter"]), row["undirected_edge_count"]))
    edges = candidate_edges[int(selected["parameter"])]
    selected_stats = graph_stats(50, edges, matrix)

    edge_rows = []
    for source, target in sorted(edges):
        for directed_source, directed_target in ((source, target), (target, source)):
            edge_rows.append({
                "source_index": directed_source,
                "target_index": directed_target,
                "source_pathway": names[directed_source],
                "target_pathway": names[directed_target],
                "edge_weight": matrix[directed_source][directed_target],
                "shared_gene_count": intersections[directed_source][directed_target],
                "source_gene_count": len(pathways[directed_source]["genes"]),
                "target_gene_count": len(pathways[directed_target]["genes"]),
            })
    write_csv(args.output_dir / "pathway_edges.csv", list(edge_rows[0]), edge_rows)
    write_csv(
        args.output_dir / "pathway_nodes.csv",
        ["node_index", "pathway", "gene_count", "degree"],
        [{"node_index": index, "pathway": pathway["name"], "gene_count": len(pathway["genes"]), "degree": selected_stats["degrees"][index]} for index, pathway in enumerate(pathways)],
    )
    print(f"Wrote {len(edge_rows)} directed edges with Top-K={int(selected['parameter'])}.")


if __name__ == "__main__":
    main()
