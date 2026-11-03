"""Tests for graph store provider."""

import pytest
from synap.providers.graph import (
    GraphStoreInterface,
    Node,
    Edge,
    TraversalPattern,
    GraphQuery,
    GraphResult,
    get_graph_store,
    reset_graph_store,
    Neo4jAdapter,
)


def test_interface_cannot_instantiate():
    """Verify GraphStoreInterface is abstract."""
    with pytest.raises(TypeError):
        GraphStoreInterface()


def test_adapter_inherits_interface():
    """Verify Neo4jAdapter inherits from GraphStoreInterface."""
    assert issubclass(Neo4jAdapter, GraphStoreInterface)


def test_node_model():
    """Verify Node model."""
    node = Node(
        id="node1",
        node_type="Person",
        properties={"name": "Alice"}
    )
    assert node.id == "node1"
    assert node.node_type == "Person"
    assert node.properties == {"name": "Alice"}


def test_edge_model():
    """Verify Edge model."""
    edge = Edge(
        id="edge1",
        from_id="node1",
        to_id="node2",
        edge_type="KNOWS",
        properties={"since": 2020}
    )
    assert edge.id == "edge1"
    assert edge.from_id == "node1"
    assert edge.to_id == "node2"
    assert edge.edge_type == "KNOWS"


def test_traversal_pattern_model():
    """Verify TraversalPattern model."""
    pattern = TraversalPattern(
        edge_types=["KNOWS", "WORKS_WITH"],
        max_depth=3,
        direction="out",
        limit=50
    )
    assert pattern.edge_types == ["KNOWS", "WORKS_WITH"]
    assert pattern.max_depth == 3
    assert pattern.direction == "out"
    assert pattern.limit == 50


def test_graph_query_model():
    """Verify GraphQuery model."""
    query = GraphQuery(
        start_node_id="node1",
        namespace="instance_1"
    )
    assert query.start_node_id == "node1"
    assert query.namespace == "instance_1"


def test_graph_result_model():
    """Verify GraphResult model."""
    result = GraphResult(
        nodes=[],
        edges=[],
        paths=[["node1", "node2"]]
    )
    assert result.nodes == []
    assert result.edges == []
    assert result.paths == [["node1", "node2"]]


# Integration Tests

@pytest.fixture
def graph():
    """Get graph store for testing."""
    reset_graph_store()
    store = get_graph_store({
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "test",
        "database": "neo4j"
    })
    yield store
    reset_graph_store()


@pytest.fixture
def namespace(graph):
    """Create test namespace."""
    ns = "test_ns"
    graph.create_namespace(ns)
    yield ns
    graph.delete_namespace(ns)


@pytest.mark.integration
def test_add_and_get_node(graph, namespace):
    """Test adding and retrieving a node."""
    node_id = graph.add_node(namespace, "Person", {"name": "Bob", "age": 30})
    node = graph.get_node(namespace, node_id)

    assert node is not None
    assert node.id == node_id
    assert node.node_type == "Person"
    assert node.properties["name"] == "Bob"
    assert node.properties["age"] == 30


@pytest.mark.integration
def test_update_node(graph, namespace):
    """Test updating node properties."""
    node_id = graph.add_node(namespace, "Person", {"name": "Alice"})
    graph.update_node(namespace, node_id, {"age": 25, "city": "NYC"})
    
    node = graph.get_node(namespace, node_id)
    assert node.properties["name"] == "Alice"
    assert node.properties["age"] == 25
    assert node.properties["city"] == "NYC"


@pytest.mark.integration
def test_delete_node(graph, namespace):
    """Test deleting a node."""
    node_id = graph.add_node(namespace, "Person", {"name": "Charlie"})
    graph.delete_node(namespace, node_id)
    
    node = graph.get_node(namespace, node_id)
    assert node is None


@pytest.mark.integration
def test_add_edge(graph, namespace):
    """Test adding an edge between nodes."""
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B"})
    edge_id = graph.add_edge(namespace, a, b, "KNOWS", {"since": 2020})

    edge = graph.get_edge(namespace, edge_id)
    assert edge is not None
    assert edge.from_id == a
    assert edge.to_id == b
    assert edge.edge_type == "KNOWS"
    assert edge.properties["since"] == 2020


@pytest.mark.integration
def test_delete_edge(graph, namespace):
    """Test deleting an edge."""
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B"})
    edge_id = graph.add_edge(namespace, a, b, "KNOWS")
    
    graph.delete_edge(namespace, edge_id)
    edge = graph.get_edge(namespace, edge_id)
    assert edge is None


@pytest.mark.integration
def test_get_neighbors_out(graph, namespace):
    """Test getting outgoing neighbors."""
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B"})
    c = graph.add_node(namespace, "Person", {"name": "C"})
    
    graph.add_edge(namespace, a, b, "KNOWS")
    graph.add_edge(namespace, a, c, "KNOWS")
    graph.add_edge(namespace, b, a, "KNOWS")  # Should not appear in out neighbors
    
    neighbors = graph.get_neighbors(namespace, a, edge_type="KNOWS", direction="out")
    assert len(neighbors) == 2
    names = {n.properties["name"] for n in neighbors}
    assert names == {"B", "C"}


@pytest.mark.integration
def test_get_neighbors_in(graph, namespace):
    """Test getting incoming neighbors."""
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B"})
    c = graph.add_node(namespace, "Person", {"name": "C"})
    
    graph.add_edge(namespace, b, a, "KNOWS")
    graph.add_edge(namespace, c, a, "KNOWS")
    graph.add_edge(namespace, a, b, "KNOWS")  # Should not appear in in neighbors
    
    neighbors = graph.get_neighbors(namespace, a, edge_type="KNOWS", direction="in")
    assert len(neighbors) == 2
    names = {n.properties["name"] for n in neighbors}
    assert names == {"B", "C"}


@pytest.mark.integration
def test_traverse(graph, namespace):
    """Test graph traversal with pattern."""
    # Create chain: A -> B -> C
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B"})
    c = graph.add_node(namespace, "Person", {"name": "C"})
    graph.add_edge(namespace, a, b, "KNOWS")
    graph.add_edge(namespace, b, c, "KNOWS")

    result = graph.traverse(namespace, a, TraversalPattern(
        edge_types=["KNOWS"],
        max_depth=2,
        direction="out"
    ))

    # Should find B and C
    assert len(result.nodes) >= 2
    names = {n.properties["name"] for n in result.nodes}
    assert "B" in names
    assert "C" in names


@pytest.mark.integration
def test_traverse_with_filters(graph, namespace):
    """Test traversal with node filters."""
    a = graph.add_node(namespace, "Person", {"name": "A"})
    b = graph.add_node(namespace, "Person", {"name": "B", "age": 25})
    c = graph.add_node(namespace, "Person", {"name": "C", "age": 30})
    
    graph.add_edge(namespace, a, b, "KNOWS")
    graph.add_edge(namespace, a, c, "KNOWS")

    result = graph.traverse(namespace, a, TraversalPattern(
        edge_types=["KNOWS"],
        max_depth=1,
        direction="out",
        node_filters={"age": 25}
    ))

    # Should only find B
    assert len(result.nodes) == 1
    assert result.nodes[0].properties["name"] == "B"


@pytest.mark.integration
def test_query_by_node_type(graph, namespace):
    """Test querying nodes by type."""
    graph.add_node(namespace, "Person", {"name": "Alice"})
    graph.add_node(namespace, "Person", {"name": "Bob"})
    graph.add_node(namespace, "Document", {"title": "Doc1"})

    query = GraphQuery(
        namespace=namespace,
        start_node_type="Person"
    )
    result = graph.query(query)

    assert len(result.nodes) >= 2
    for node in result.nodes:
        assert node.node_type == "Person"


@pytest.mark.integration
def test_namespace_isolation(graph):
    """Test that namespaces are isolated from each other."""
    ns1 = "namespace_1"
    ns2 = "namespace_2"
    
    graph.create_namespace(ns1)
    graph.create_namespace(ns2)
    
    # Add node to ns1
    node_id = graph.add_node(ns1, "Person", {"name": "Alice"})
    
    # Should exist in ns1
    node = graph.get_node(ns1, node_id)
    assert node is not None
    
    # Should not exist in ns2
    node = graph.get_node(ns2, node_id)
    assert node is None

    # Cleanup
    graph.delete_namespace(ns1)
    graph.delete_namespace(ns2)


@pytest.mark.integration
def test_lookups_avoid_full_scans(graph, namespace):
    """PR-1 (T1.2/T1.5): the memory reverse-lookup and node id lookup must NOT
    plan as full scans. The old label-less queries planned as
    DirectedAllRelationshipsScan / AllNodesScan (cross-tenant, unbounded); the
    rewrites anchor on the namespace-indexed :SynapNode. This guards against a
    regression back to a scan. See Neo4j_Implementation_Improvement_Design_v1.md.
    """
    import json

    a = graph.add_node(namespace, "Person", {"name": "A", "canonical_name": "A"})
    b = graph.add_node(namespace, "Project", {"name": "B", "canonical_name": "B"})
    e = graph.add_edge(namespace, a, b, "WORKS_ON", {"role": "lead"})
    graph.update_edge(namespace, e, {"memory_contents": json.dumps([{"memory_id": "mem-1"}])})

    def _plan_ops(cypher, **params):
        driver = graph._get_driver()
        with driver.session(database=graph._database) as session:
            res = session.run("PROFILE " + cypher, params)
            list(res)
            prof = res.consume().profile
        ops = []

        def _walk(p):
            if not p:
                return
            ops.append(p.get("operatorType", ""))
            for c in (p.get("children") or []):
                _walk(c)

        _walk(prof)
        return ops

    # T1.2: namespace-anchored memory reverse lookup — must not scan all rels
    rev = _plan_ops(
        "MATCH (a:SynapNode {_namespace: $ns})-[r]->() "
        "WHERE r._namespace = $ns AND r.memory_contents CONTAINS $mid RETURN r.id",
        ns=namespace, mid="mem-1",
    )
    assert not any("AllRelationshipsScan" in o for o in rev), \
        f"reverse lookup regressed to a full relationship scan: {rev}"

    # T1.5: node id lookup — must not scan all nodes
    node_ops = _plan_ops(
        "MATCH (n:SynapNode {_namespace: $ns, id: $id}) RETURN n",
        ns=namespace, id=a,
    )
    assert not any("AllNodesScan" in o for o in node_ops), \
        f"node id lookup regressed to a full node scan: {node_ops}"
