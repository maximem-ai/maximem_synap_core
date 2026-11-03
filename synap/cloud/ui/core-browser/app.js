"use strict";
// Synap Core memory browser. Read-only. No framework, no build step.
// Every value from the server is put on the page as text, never as HTML.
// The look follows the client dashboard (same tokens, layout and table style).

const API = "/browser/api";
const TOKEN_KEY = "synap-core-token";
const app = document.getElementById("app");
const modeBadge = document.getElementById("mode");
let overview = null;
let renderId = 0;

// ---------------------------------------------------------------- helpers

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// Replace an element's children, skipping nulls and flattening nested lists. The DOM's own
// replaceChildren would print "null" for a missing node.
function show(el, ...nodes) {
  el.replaceChildren(...nodes.flat(Infinity).filter((n) => n != null && n !== false && n !== ""));
}

// Icons are the same Lucide shapes the client dashboard uses, drawn as SVG elements.
const SVG_NS = "http://www.w3.org/2000/svg";
const ICONS = {
  database: [["ellipse", { cx: 12, cy: 5, rx: 9, ry: 3 }], ["path", { d: "M3 5V19A9 3 0 0 0 21 19V5" }], ["path", { d: "M3 12A9 3 0 0 0 21 12" }]],
  users: [["path", { d: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" }], ["circle", { cx: 9, cy: 7, r: 4 }], ["path", { d: "M22 21v-2a4 4 0 0 0-3-3.87" }], ["path", { d: "M16 3.13a4 4 0 0 1 0 7.75" }]],
  user: [["path", { d: "M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" }], ["circle", { cx: 12, cy: 7, r: 4 }]],
  building: [["rect", { x: 4, y: 2, width: 16, height: 20, rx: 2 }], ["path", { d: "M9 22v-4h6v4" }], ["path", { d: "M8 6h.01M16 6h.01M12 6h.01M12 10h.01M12 14h.01M16 10h.01M16 14h.01M8 10h.01M8 14h.01" }]],
  file: [["path", { d: "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" }], ["path", { d: "M14 2v4a2 2 0 0 0 2 2h4" }], ["path", { d: "M10 9H8M16 13H8M16 17H8" }]],
  search: [["circle", { cx: 11, cy: 11, r: 8 }], ["path", { d: "m21 21-4.3-4.3" }]],
  chart: [["path", { d: "M3 3v16a2 2 0 0 0 2 2h16" }], ["path", { d: "M18 17V9" }], ["path", { d: "M13 17V5" }], ["path", { d: "M8 17v-3" }]],
  activity: [["path", { d: "M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" }]],
  copy: [["rect", { x: 8, y: 8, width: 14, height: 14, rx: 2 }], ["path", { d: "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" }]],
  check: [["path", { d: "M20 6 9 17l-5-5" }]],
  alert: [["circle", { cx: 12, cy: 12, r: 10 }], ["path", { d: "M12 8v4M12 16h.01" }]],
  left: [["path", { d: "m15 18-6-6 6-6" }]],
  right: [["path", { d: "m9 18 6-6-6-6" }]],
  refresh: [["path", { d: "M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" }], ["path", { d: "M21 3v5h-5" }], ["path", { d: "M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" }], ["path", { d: "M8 16H3v5" }]],
};

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", class: "icon", "aria-hidden": "true" })) svg.setAttribute(k, v);
  for (const [tag, attrs] of ICONS[name] || []) {
    const shape = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) shape.setAttribute(k, v);
    svg.append(shape);
  }
  return svg;
}

for (const slot of document.querySelectorAll("[data-icon]")) slot.replaceWith(icon(slot.dataset.icon));

const DATE_FORMAT = { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" };

function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("en-US", DATE_FORMAT);
}

function num(n) {
  return Number(n || 0).toLocaleString();
}

function percent(x) {
  return x == null ? "" : Math.round(x * 100) + "%";
}

function getToken() {
  try { return sessionStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}

function setToken(value) {
  try { sessionStorage.setItem(TOKEN_KEY, value); } catch { /* private window: the token lasts until reload */ }
}

class AuthNeeded extends Error {}

async function api(path, params) {
  const url = new URL(API + path, location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== "" && v != null) url.searchParams.set(k, v);
  }
  const headers = {};
  const token = getToken();
  if (token) headers.Authorization = "Bearer " + token;
  const res = await fetch(url, { headers });
  if (res.status === 401) throw new AuthNeeded();
  if (!res.ok) {
    let message = res.statusText || "Request failed";
    try {
      const body = await res.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (body.detail && body.detail.message) message = body.detail.message;
    } catch { /* keep the status text */ }
    const err = new Error(message);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "/overview";
  const [path, query = ""] = raw.split("?");
  return {
    parts: path.split("/").filter(Boolean).map(decodeURIComponent),
    query: Object.fromEntries(new URLSearchParams(query)),
  };
}

function hashFor(path, query) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(query || {})) if (v !== "" && v != null) q.set(k, v);
  const s = q.toString();
  return "#" + path + (s ? "?" + s : "");
}

function link(text, href, cls) {
  return h("a", { href, class: cls }, text);
}

function memoryLink(id) {
  return "#/memories/" + encodeURIComponent(id);
}

function entityLink(kind, id) {
  return "#/entities/" + kind + "/" + encodeURIComponent(id);
}

function empty(text) {
  return h("div", { class: "empty" }, text);
}

function back(text, href) {
  return h("a", { class: "back", href }, icon("left"), text);
}

function titleCase(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, " ") : "";
}

function scopeLabel(scope) {
  return scope === "install" ? "Install" : titleCase(scope);
}

function ownerId(item) {
  return item.user_id || item.customer_id || "";
}

function ownerCell(item) {
  if (item.scope === "install") return h("span", { class: "mono muted" }, "Install");
  const kind = item.user_id ? "user" : "customer";
  return link(ownerId(item), entityLink(kind, ownerId(item)), "mono");
}

function field(iconName, input) {
  return h("div", { class: "field" }, icon(iconName), input);
}

function pager(page, totalPages, noun, total, onPage) {
  return h("div", { class: "foot" },
    h("span", null, num(total) + " total " + noun),
    h("div", { class: "pager" },
      h("button", { type: "button", "aria-label": "Previous page", disabled: page <= 1, onclick: () => onPage(page - 1) }, icon("left")),
      h("span", { class: "of" }, "Page", h("span", { class: "cur" }, page), "of " + num(totalPages)),
      h("button", { type: "button", "aria-label": "Next page", disabled: page >= totalPages, onclick: () => onPage(page + 1) }, icon("right"))));
}

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function openRow(row, href) {
  const open = () => { location.hash = href; };
  row.addEventListener("click", (e) => { if (!e.target.closest("a")) open(); });
  row.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  return row;
}

// ---------------------------------------------------------------- pages

async function memoriesPage(q, rid) {
  const state = {
    page: Math.max(1, parseInt(q.page, 10) || 1),
    search: q.search || "", type: q.type || "", entity: q.entity || "", status: q.status || "active",
  };
  const results = h("div", { class: "results" });
  let typeList = [];

  const sync = () => history.replaceState(null, "", hashFor("/memories", {
    page: state.page > 1 ? state.page : "", search: state.search, type: state.type,
    entity: state.entity, status: state.status === "active" ? "" : state.status,
  }));
  const restart = () => { state.page = 1; load(); };

  const search = h("input", { type: "search", placeholder: "Search memory content...", value: state.search, "aria-label": "Search memory content" });
  search.addEventListener("input", debounce(() => { state.search = search.value; restart(); }, 300));
  const entity = h("input", { type: "search", placeholder: "Filter by user or customer ID...", value: state.entity, "aria-label": "Filter by user or customer id" });
  entity.addEventListener("input", debounce(() => { state.entity = entity.value; restart(); }, 300));
  const typeSelect = h("select", { "aria-label": "Memory type" }, h("option", { value: "" }, "All Types"));
  typeSelect.addEventListener("change", () => { state.type = typeSelect.value; restart(); });
  const status = h("select", { "aria-label": "Memory status" },
    [["active", "Active"], ["retired", "Retired"], ["all", "Active and retired"]].map(([v, label]) => h("option", { value: v, selected: v === state.status }, label)));
  status.addEventListener("change", () => { state.status = status.value; restart(); });
  const clear = h("button", { type: "button", onclick: () => {
    Object.assign(state, { search: "", type: "", entity: "", status: "active" });
    search.value = ""; entity.value = ""; status.value = "active"; restart();
  } }, "Clear");

  function fillTypes(byType) {
    if (!state.type && byType.length) typeList = byType;
    const known = typeList.slice();
    if (state.type && !known.some((t) => t.memory_type === state.type)) known.push({ memory_type: state.type, count: 0 });
    show(typeSelect, h("option", { value: "" }, "All Types"),
      known.map((t) => h("option", { value: t.memory_type, selected: t.memory_type === state.type }, titleCase(t.memory_type) + " (" + num(t.count) + ")")));
  }

  async function load() {
    sync();
    let data;
    try {
      data = await api("/memories", { page: state.page, page_size: 25, search: state.search, type: state.type, entity: state.entity, status: state.status });
    } catch (err) {
      if (rid !== renderId) return;
      if (err instanceof AuthNeeded) return askForToken();
      show(results, h("div", { class: "error" }, err.message));
      return;
    }
    if (rid !== renderId) return;
    fillTypes(data.by_type);
    const filtered = state.search || state.type || state.entity;
    show(results,
      data.items.length === 0
        ? h("div", { class: "card" }, empty(filtered
          ? "No memory matches these filters."
          : "Nothing stored yet. Add a memory with the SDK, then reload this page."))
        : h("div", { class: "card" }, h("table", null,
          h("thead", null, h("tr", null,
            h("th", null, "Type"), h("th", null, "Scope"), h("th", { class: "hide-sm" }, "Entity ID"),
            h("th", null, "Content"), h("th", { class: "hide-sm" }, "Confidence"), h("th", { class: "hide-sm" }, "Created"))),
          h("tbody", null, data.items.map((m) => openRow(h("tr", { class: "link", tabindex: "0" },
            h("td", null, h("span", { class: "pill" }, titleCase(m.memory_type))),
            h("td", null, h("span", { class: "pill" }, scopeLabel(m.scope))),
            h("td", { class: "hide-sm" }, ownerCell(m)),
            h("td", null, h("div", { class: "clip" }, m.content), m.status === "retired" ? h("span", { class: "badge warn" }, "retired") : null),
            h("td", { class: "hide-sm" }, percent(m.confidence)),
            h("td", { class: "hide-sm" }, when(m.created_at))), memoryLink(m.id)))))),
      data.total > 0 ? pager(data.page, data.total_pages, "memories", data.total, (p) => { state.page = p; load(); window.scrollTo(0, 0); }) : "");
  }

  show(app,
    h("div", { class: "head" }, h("div", null,
      h("h1", null, "Memories"),
      h("p", { class: "sub" }, overview
        ? num(overview.memories_active) + " active, " + num(overview.memories_retired) + " retired. Read-only view of what this install has stored."
        : "Read-only view of what this install has stored."))),
    h("div", { class: "bar" }, field("search", search), field("user", entity), typeSelect, status, clear),
    results);
  fillTypes([]);
  await load();
}

async function memoryDetailPage(id, rid) {
  const m = await api("/memories/" + encodeURIComponent(id));
  if (rid !== renderId) return;
  const meta = [
    ["Type", titleCase(m.memory_type) + (m.category ? " / " + m.category : "") + (m.subcategory ? " / " + m.subcategory : "")],
    ["Scope", scopeLabel(m.scope)],
    ["Belongs to", m.scope === "install" ? "the whole install" : ownerCell(m)],
    ["Confidence", percent(m.confidence)],
    ["Created", when(m.created_at)],
    ["Updated", when(m.updated_at)],
    ["Time range", m.temporal_category + (m.valid_from ? ", from " + when(m.valid_from) : "") + (m.valid_until ? " until " + when(m.valid_until) : "")],
    ["Source document", m.source_document_id],
    ["Memory id", h("span", { class: "mono" }, m.id)],
  ].filter(([, v]) => v);
  show(app,
    back("Back to memories", "#/memories"),
    h("div", { class: "head" }, h("h1", null, "Memory")),
    m.status === "retired"
      ? h("div", { class: "notice" },
        "Retired" + (m.superseded_at || m.deprecated_at ? " on " + when(m.superseded_at || m.deprecated_at) : "") + ".",
        m.superseded_by ? [" Replaced by ", link("a later memory", memoryLink(m.superseded_by), "underline"), m.replacement_content ? ": " + m.replacement_content : "."] : "")
      : null,
    h("div", { class: "card" }, h("div", { class: "content" }, m.content)),
    h("h2", null, "Details"),
    h("div", { class: "card" }, h("dl", { class: "meta" }, meta.map(([k, v]) => [h("dt", null, k), h("dd", null, v)]))),
    h("h2", null, "History"),
    m.history.length === 0
      ? h("div", { class: "card" }, empty("No recorded changes for this memory."))
      : h("div", { class: "card" }, h("ul", { class: "timeline" }, m.history.map((e) => h("li", null,
        h("span", { class: "pill" }, e.action), h("span", { class: "when" }, when(e.created_at)),
        e.reason ? h("div", { class: "muted" }, e.reason) : null,
        e.previous_content ? h("div", { class: "muted" }, "Before: " + e.previous_content) : null,
        e.content ? h("div", null, e.content) : null)))));
}

async function entitiesPage(kindParam, q, rid) {
  const kinds = overview ? overview.entity_kinds : ["customer"];
  const kind = kinds.includes(kindParam) ? kindParam : kinds[0];
  const equalsCustomer = overview && overview.scoping_mode === "equals_customer";
  const label = (k) => (equalsCustomer ? "User" : k === "customer" ? "Customer" : "User");
  const state = { page: Math.max(1, parseInt(q.page, 10) || 1), search: q.search || "" };
  const results = h("div", { class: "results" });
  const sync = () => history.replaceState(null, "", hashFor("/entities/" + kind, { page: state.page > 1 ? state.page : "", search: state.search }));

  async function load() {
    sync();
    let data;
    try {
      data = await api("/entities", { kind, page: state.page, page_size: 25, search: state.search });
    } catch (err) {
      if (rid !== renderId) return;
      if (err instanceof AuthNeeded) return askForToken();
      show(results, h("div", { class: "error" }, err.message));
      return;
    }
    if (rid !== renderId) return;
    show(results,
      data.items.length === 0
        ? h("div", { class: "card" }, empty(state.search ? "No match." : "No " + label(kind).toLowerCase() + "s yet. They appear when memories are stored for them."))
        : h("div", { class: "card" }, h("table", null,
          h("thead", null, h("tr", null,
            h("th", null, label(kind)),
            kind === "user" ? h("th", { class: "hide-sm" }, "Parent customer") : null,
            h("th", null, "Memories"), h("th", { class: "hide-sm" }, "First seen"))),
          h("tbody", null, data.items.map((e) => openRow(h("tr", { class: "link", tabindex: "0" },
            h("td", { class: "mono" }, e.external_id),
            kind === "user" ? h("td", { class: "hide-sm mono" }, e.parent_customer_id || "") : null,
            h("td", null, h("span", { class: "pill" }, num(e.memories_active)), e.memories_retired ? h("span", { class: "muted" }, " +" + num(e.memories_retired) + " retired") : null),
            h("td", { class: "hide-sm" }, when(e.created_at))), entityLink(kind, e.external_id)))))),
      data.total > 0 ? pager(data.page, data.total_pages, label(kind).toLowerCase() + "s", data.total, (p) => { state.page = p; load(); window.scrollTo(0, 0); }) : "");
  }

  const search = h("input", { type: "search", placeholder: "Search " + label(kind).toLowerCase() + "s...", value: state.search, "aria-label": "Search by id or name" });
  search.addEventListener("input", debounce(() => { state.search = search.value; state.page = 1; load(); }, 300));
  show(app,
    h("div", { class: "head" }, h("div", null,
      h("h1", null, "Entities"),
      h("p", { class: "sub" }, equalsCustomer
        ? "The users this install knows. Each user is their own customer in this scoping mode."
        : "The customers and users this install knows."))),
    h("div", { class: "head" },
      kinds.length > 1
        ? h("div", { class: "seg" }, kinds.map((k) => h("a", { class: k === kind ? "on" : "", href: "#/entities/" + k }, icon(k === "customer" ? "building" : "user"), label(k))))
        : h("div", { class: "seg" }, h("a", { class: "on", href: "#/entities/" + kind }, icon("user"), label(kind))),
      h("div", { class: "bar flat" }, field("search", search),
        h("button", { type: "button", onclick: () => load() }, icon("refresh"), " Refresh"))),
    results);
  await load();
}

async function entityDetailPage(kind, id, rid) {
  const e = await api("/entities/" + encodeURIComponent(kind) + "/" + encodeURIComponent(id));
  if (rid !== renderId) return;
  const equalsCustomer = overview && overview.scoping_mode === "equals_customer";
  const noun = kind === "customer" && !equalsCustomer ? "Customer" : "User";
  const meta = [
    [noun + " id", h("span", { class: "mono" }, e.external_id)],
    ["Name", e.name],
    ["Description", e.description],
    ["Parent customer", e.parent_customer_id ? link(e.parent_customer_id, entityLink("customer", e.parent_customer_id), "mono") : null],
    ["First seen", when(e.created_at)],
    ["Last updated", when(e.updated_at)],
    ["Active memories", num(e.memories_active)],
    ["Retired memories", num(e.memories_retired)],
  ].filter(([, v]) => v);
  show(app,
    back("Back to entities", "#/entities/" + kind),
    h("div", { class: "head" }, h("h1", null, noun)),
    h("div", { class: "card" }, h("dl", { class: "meta" }, meta.map(([k, v]) => [h("dt", null, k), h("dd", null, v)]))),
    e.by_type.length ? h("div", { class: "chips" }, e.by_type.map((t) => h("span", { class: "pill" }, titleCase(t.memory_type) + " " + num(t.count)))) : null,
    h("h2", null, "Profile"),
    e.profile
      ? h("div", { class: "card" }, h("pre", null, JSON.stringify(e.profile, null, 2)))
      : h("div", { class: "card" }, empty("No profile yet. Core builds one when a conversation ends or goes idle.")),
    h("h2", null, "Latest memories"),
    e.latest_memories.length === 0
      ? h("div", { class: "card" }, empty("No active memories."))
      : h("div", { class: "card" }, h("table", null, h("tbody", null, e.latest_memories.map((m) => openRow(h("tr", { class: "link", tabindex: "0" },
        h("td", null, h("span", { class: "pill" }, titleCase(m.memory_type))),
        h("td", null, h("div", { class: "clip" }, m.content)),
        h("td", { class: "hide-sm" }, when(m.created_at))), memoryLink(m.id)))))),
    e.memories_active > 10
      ? link("See all " + num(e.memories_active) + " memories", hashFor("/memories", { entity: e.external_id }), "back")
      : null);
}

// ---------------------------------------------------------------- charts

function svgEl(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const kid of kids) if (kid != null) el.append(kid);
  return el;
}

function niceCeil(n) {
  if (n <= 4) return 4;
  const mag = Math.pow(10, Math.floor(Math.log10(n)));
  for (const step of [1, 2, 4, 5, 10]) if (n <= step * mag) return step * mag;
  return 10 * mag;
}

function shortDay(iso) {
  const d = new Date(iso + "T00:00:00Z");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

// A column chart over days. Colours come from the stylesheet, so the theme decides them.
function dayChart(points, key, noun, tone) {
  const W = 640, H = 220, L = 44, R = 8, T = 10, B = 26;
  const top = niceCeil(Math.max(0, ...points.map((p) => p[key])));
  const slot = (W - L - R) / points.length;
  const bar = Math.max(1, slot * 0.7);
  const y = (v) => T + (H - T - B) * (1 - v / top);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img",
    "aria-label": `${noun} per day, ${points.length} days, most ${Math.max(0, ...points.map((p) => p[key]))}` });
  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "chart-grid" }));
    const label = svgEl("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "chart-axis" });
    label.textContent = num(Math.round(v));
    svg.append(label);
  }
  points.forEach((p, i) => {
    const x = L + slot * i + (slot - bar) / 2;
    const rect = svgEl("rect", { x, y: y(p[key]), width: bar, height: Math.max(0, y(0) - y(p[key])), rx: 2, class: "chart-bar " + tone });
    const tip = svgEl("title"); tip.textContent = `${shortDay(p.day)}: ${num(p[key])} ${noun}`;
    rect.append(tip);
    svg.append(rect);
  });
  [0, Math.floor((points.length - 1) / 2), points.length - 1].forEach((i, n) => {
    const label = svgEl("text", { x: n === 0 ? L : n === 2 ? W - R : L + slot * i + slot / 2, y: H - 6, "text-anchor": n === 0 ? "start" : n === 2 ? "end" : "middle", class: "chart-axis" });
    label.textContent = shortDay(points[i].day);
    svg.append(label);
  });
  return svg;
}

// Horizontal bars with a label and a count, for a short list.
function rowBars(items, labelOf) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return h("div", { class: "rowbars" }, items.map((i) => h("div", { class: "rowbar" },
    h("span", { class: "rb-label" }, labelOf(i)),
    svgEl("svg", { viewBox: "0 0 100 8", preserveAspectRatio: "none", class: "rb-track", "aria-hidden": "true" },
      svgEl("rect", { x: 0, y: 0, width: 100, height: 8, rx: 4, class: "rb-bg" }),
      svgEl("rect", { x: 0, y: 0, width: Math.max(1.5, (i.count / max) * 100), height: 8, rx: 4, class: "rb-fill" })),
    h("span", { class: "rb-n" }, num(i.count)))));
}

const ACTION_LABELS = { CREATE: "Created", ENRICH: "Enriched", COALESCE: "Merged", SUPERSEDE: "Superseded", UPDATE: "Updated", DELETE: "Deleted", NONE: "Unchanged" };
const actionLabel = (a) => ACTION_LABELS[a] || titleCase(String(a || "").toLowerCase());

function outcomePills(outcomes) {
  const entries = Object.entries(outcomes || {}).sort((a, b) => b[1] - a[1]);
  return h("div", { class: "chips" }, entries.map(([a, n]) => h("span", { class: "pill" }, actionLabel(a) + " " + num(n))));
}

function statCard(title, value, sub) {
  return h("div", { class: "stat" }, h("span", { class: "stat-title" }, title), h("b", null, value), h("span", { class: "muted" }, sub));
}

function panelCard(title, sub, body, extra) {
  return h("section", { class: "pcard" },
    h("div", { class: "pcard-head" }, h("div", null, h("h3", null, title), sub ? h("p", { class: "muted" }, sub) : null), extra),
    body);
}

// ---------------------------------------------------------------- overview

async function overviewPage(q, rid) {
  const days = [7, 30, 90].includes(parseInt(q.days, 10)) ? parseInt(q.days, 10) : 30;
  const [a, recent, cfg] = await Promise.all([api("/analytics", { days }), api("/requests", { page_size: 5 }), api("/config").catch(() => null)]);
  if (rid !== renderId) return;
  const strict = overview.entity_kinds.includes("user");
  const range = h("div", { class: "seg" }, [7, 30, 90].map((d) => h("a", { class: d === days ? "on" : "", href: hashFor("/overview", { days: d }) }, d + "d")));
  const nothing = a.memories_active + a.memories_retired === 0;
  show(app,
    h("div", { class: "head" }, h("div", null, h("h1", null, "Overview"),
      h("p", { class: "sub" }, "What this install has stored and how it has been written to. Days are UTC. Read-only.")), range),
    cfg && cfg.starter ? h("div", { class: "notice" }, "Memory is running on the starter config, so nothing is tuned to your app yet. ", h("a", { href: "#/config" }, "See how to make it yours"), ".") : null,
    nothing ? h("div", { class: "card" }, empty("Nothing stored yet. Add a memory with the SDK, then reload this page.")) : null,
    h("div", { class: "stats" },
      statCard("Active memories", num(a.memories_active), "+" + num(a.memories_created) + " in the last " + days + " days"),
      statCard("Write requests", num(a.requests), "in the last " + days + " days"),
      statCard(strict ? "Customers" : "Users", num(overview.customers), strict ? num(overview.users) + " users under them" : "each user is their own customer"),
      statCard("Retired", num(a.memories_retired), "replaced by a newer fact")),
    h("div", { class: "grid2" },
      panelCard("Memories stored", "New memories per day, last " + days + " days.", dayChart(a.per_day, "memories", "memories", "")),
      panelCard("Write requests", "Requests that stored text, per day.", dayChart(a.per_day, "requests", "requests", "alt"))),
    h("div", { class: "grid2" },
      panelCard("What the writes did", "Outcome of each memory a write touched, last " + days + " days.",
        a.outcomes.length ? rowBars(a.outcomes.map((o) => ({ label: actionLabel(o.action), count: o.count })), (i) => i.label) : empty("No writes in this period.")),
      panelCard("Memories by type", "Active memories.",
        a.by_type.length ? rowBars(a.by_type.map((t) => ({ label: titleCase(t.memory_type), count: t.count })), (i) => i.label) : empty("None yet."))),
    h("div", { class: "grid2" },
      panelCard("Most memories", "The " + (strict ? "customers and users" : "users") + " with the most active memories.",
        a.top_entities.length
          ? h("table", null, h("tbody", null, a.top_entities.map((t) => openRow(h("tr", { class: "link", tabindex: "0" },
            h("td", { class: "mono" }, t.external_id), h("td", { class: "num" }, num(t.memories_active))), entityLink(t.kind, t.external_id)))))
          : empty("None yet.")),
      panelCard("Latest requests", null,
        recent.items.length
          ? h("table", null, h("tbody", null, recent.items.map((r) => openRow(h("tr", { class: "link", tabindex: "0" },
            h("td", null, when(r.started_at)), h("td", { class: "mono" }, r.entity || ""),
            h("td", { class: "num" }, num(r.actions) + " memories")), requestLink(r.request_id)))))
          : empty("No writes yet."),
        link("See all", "#/requests", "back"))));
}

// ---------------------------------------------------------------- requests

function requestLink(id) {
  return "#/requests/" + encodeURIComponent(id);
}

async function requestsPage(q, rid) {
  const state = { page: Math.max(1, parseInt(q.page, 10) || 1), entity: q.entity || "", days: q.days || "" };
  const results = h("div", { class: "results" });
  const sync = () => history.replaceState(null, "", hashFor("/requests", { page: state.page > 1 ? state.page : "", entity: state.entity, days: state.days }));
  const restart = () => { state.page = 1; load(); };

  async function load() {
    sync();
    let data;
    try {
      data = await api("/requests", { page: state.page, page_size: 25, entity: state.entity, days: state.days });
    } catch (err) {
      if (rid !== renderId) return;
      if (err instanceof AuthNeeded) return askForToken();
      show(results, h("div", { class: "error" }, err.message));
      return;
    }
    if (rid !== renderId) return;
    show(results,
      data.items.length === 0
        ? h("div", { class: "card" }, empty(state.entity || state.days ? "No request matches these filters." : "No writes yet. Store a memory with the SDK, then reload this page."))
        : h("div", { class: "card" }, h("table", null,
          h("thead", null, h("tr", null,
            h("th", null, "Started"), h("th", null, "Request"), h("th", { class: "hide-sm" }, "Entity"),
            h("th", null, "Memories touched"), h("th", { class: "hide-sm" }, "What it did"))),
          h("tbody", null, data.items.map((r) => openRow(h("tr", { class: "link", tabindex: "0" },
            h("td", null, when(r.started_at)),
            h("td", { class: "mono" }, r.request_id),
            h("td", { class: "hide-sm mono" }, r.entity ? r.entity + (r.entities > 1 ? " +" + (r.entities - 1) : "") : ""),
            h("td", null, num(r.actions)),
            h("td", { class: "hide-sm" }, outcomePills(r.outcomes))), requestLink(r.request_id)))))),
      data.total > 0 ? pager(data.page, data.total_pages, "requests", data.total, (p) => { state.page = p; load(); window.scrollTo(0, 0); }) : "");
  }

  const entity = h("input", { type: "search", placeholder: "Filter by user or customer ID...", value: state.entity, "aria-label": "Filter by user or customer id" });
  entity.addEventListener("input", debounce(() => { state.entity = entity.value; restart(); }, 300));
  const range = h("select", { "aria-label": "Time range" },
    [["", "All time"], ["7", "Last 7 days"], ["30", "Last 30 days"], ["90", "Last 90 days"]].map(([v, label]) => h("option", { value: v, selected: v === state.days }, label)));
  range.addEventListener("change", () => { state.days = range.value; restart(); });
  const clear = h("button", { type: "button", onclick: () => { state.entity = ""; state.days = ""; entity.value = ""; range.value = ""; restart(); } }, "Clear");

  show(app,
    h("div", { class: "head" }, h("div", null, h("h1", null, "Requests"),
      h("p", { class: "sub" }, "Memory writes: each request that stored text, and what it did to the memories. Core does not keep a log of other API calls."))),
    h("div", { class: "bar" }, field("user", entity), range, clear),
    results);
  await load();
}

async function requestDetailPage(id, rid) {
  const r = await api("/requests/" + encodeURIComponent(id));
  if (rid !== renderId) return;
  const meta = [
    ["Request id", h("span", { class: "mono" }, r.request_id)],
    ["Started", when(r.started_at)],
    ["Finished", r.ended_at && r.ended_at !== r.started_at ? when(r.ended_at) : null],
    ["Entity", r.entity ? link(r.entity, entityLink(overview.entity_kinds.includes("user") ? "user" : "customer", r.entity), "mono") : null],
    ["Memories touched", num(r.actions)],
    ["What it did", outcomePills(r.outcomes)],
  ].filter(([, v]) => v);
  show(app,
    back("Back to requests", "#/requests"),
    h("div", { class: "head" }, h("h1", null, "Request")),
    h("div", { class: "card" }, h("dl", { class: "meta" }, meta.map(([k, v]) => [h("dt", null, k), h("dd", null, v)]))),
    h("h2", null, "Memories it touched"),
    r.truncated ? h("div", { class: "notice" }, "This request touched " + num(r.actions) + " memories. The first " + num(r.items.length) + " are shown.") : null,
    h("div", { class: "card" }, h("table", null,
      h("thead", null, h("tr", null, h("th", null, "Action"), h("th", { class: "hide-sm" }, "Type"), h("th", null, "Memory"))),
      h("tbody", null, r.items.map((i) => openRow(h("tr", { class: "link", tabindex: "0" },
        h("td", null, h("span", { class: "pill" }, actionLabel(i.action))),
        h("td", { class: "hide-sm" }, i.memory_type ? h("span", { class: "pill" }, titleCase(i.memory_type)) : ""),
        h("td", null,
          h("div", { class: "clip" }, i.content || ""),
          i.previous_content ? h("div", { class: "muted clip" }, "Before: " + i.previous_content) : null,
          i.reason ? h("div", { class: "muted clip" }, i.reason) : null)), memoryLink(i.memory_id)))))));
}

// ---------------------------------------------------------------- config

const QUICKSTART_YAML = "storage_architecture:\n  scoping:\n    user_context_isolation: equals_customer\n";
const BUSINESS_YAML = "storage_architecture:\n  scoping:\n    user_context_isolation: strict\n";
const MACA_SNIPPET = "job = await sdk.maca.generate(file_path=\"use-case.md\", user_relationship=\"b2c\")\njob = await sdk.maca.wait_for_completion(job.job_id)\n";

function copyButton(getText) {
  const label = h("span", null, "Copy");
  const button = h("button", { type: "button", class: "ghost", onclick: async () => {
    try { await navigator.clipboard.writeText(getText()); label.textContent = "Copied"; }
    catch { label.textContent = "Copy failed"; }
    setTimeout(() => { label.textContent = "Copy"; }, 1500);
  } }, icon("copy"), label);
  return button;
}

function codeBlock(title, code, note) {
  return h("div", { class: "codebox" },
    h("div", { class: "codebox-head" }, h("span", null, title), copyButton(() => code)),
    h("pre", null, code),
    note ? h("p", { class: "muted codebox-note" }, note) : null);
}

function stepCard(n, title, ...body) {
  return h("li", { class: "step" }, h("span", { class: "step-n" }, n), h("div", { class: "step-body" }, h("h3", null, title), body));
}

function modeCards() {
  return h("div", { class: "grid2 tight" },
    h("div", { class: "choice" }, h("b", null, "equals_customer"), h("span", { class: "muted" }, "A consumer app. Each user is their own customer, so your app sends a user id and nothing else.")),
    h("div", { class: "choice" }, h("b", null, "strict"), h("span", { class: "muted" }, "A business app. Users sit under a customer, so every call sends a customer id and a user id.")));
}

function setupSteps(c, hasFile) {
  const dir = c.path.replace(/memory\.yml$/, "") || "config/";
  return h("ol", { class: "steps" },
    stepCard(1, "Choose how your users are scoped", h("p", { class: "muted" }, "Decide this before you store anything. It cannot be changed once memories exist."), modeCards()),
    stepCard(2, hasFile ? "Edit the file" : "Create the file",
      h("p", { class: "muted" }, "Save it as ", h("span", { class: "mono" }, c.path), " in the folder you mounted into the server. This is all a config needs to start; everything else uses the built-in defaults."),
      h("div", { class: "grid2 tight" }, codeBlock("Consumer app", QUICKSTART_YAML), codeBlock("Business app", BUSINESS_YAML)),
      c.samples.length ? h("p", { class: "muted" }, "Or start from a sample already in your ", h("span", { class: "mono" }, dir + "samples/"), " folder: ", c.samples.join(", "), ".") : null),
    stepCard(3, "Save, then reload this page",
      h("p", { class: "muted" }, "The server reads the file again when it changes. If the new file has a mistake it keeps the last good one and logs the error; a mistake at start stops the server and names the line. This page shows the result.")),
    stepCard(4, "Or let MACA write it for you (optional)",
      h("p", { class: "muted" }, "Describe your use case in a markdown file and Synap Cloud designs the config. It needs a Synap Cloud account with credits and a management key in ", h("span", { class: "mono" }, "SYNAP_CLOUD_API_KEY"), ". Core saves the result as ", h("span", { class: "mono" }, dir + "memory.generated.yml"), ". It never replaces your ", h("span", { class: "mono" }, "memory.yml"), ": review it, then rename it."),
      codeBlock("Python", MACA_SNIPPET, "Core docs: MACA plugin.")));
}

function sourceView(text) {
  const lines = text.replace(/\n$/, "").split("\n");
  return h("div", { class: "code" },
    h("div", { class: "code-head" }, h("span", { class: "mono" }, lines.length + (lines.length === 1 ? " line" : " lines")), copyButton(() => text)),
    h("div", { class: "code-body" }, h("pre", { class: "gutter", "aria-hidden": "true" }, lines.map((_, i) => i + 1).join("\n")), h("pre", { class: "src" }, text)));
}

async function configPage(rid) {
  const c = await api("/config");
  if (rid !== renderId) return;
  const head = h("div", { class: "head" }, h("div", null, h("h1", null, "Config"),
    h("p", { class: "sub" }, "How memory behaves on this install. The server reads one YAML file; this page only shows it.")));

  if (!c.exists) {
    show(app, head,
      h("div", { class: "hero" }, icon("file"),
        h("div", null, h("h2", null, "No memory config yet"),
          h("p", { class: "muted" }, "The server is running on its built-in defaults, so memories are stored and retrieved, but nothing is tuned to your app. Four steps to add one."))),
      setupSteps(c, false));
    return;
  }

  const state = c.valid === false ? "Has problems" : c.valid ? "Accepted" : "Not checked";
  const starter = c.starter
    ? h("div", { class: "hero" }, icon("file"),
      h("div", null, h("h2", null, "You are on the starter config"),
        h("p", { class: "muted" }, "This is the small sample that ships with Core. It only decides how users are kept apart (a consumer app: each user is their own customer); everything else runs on the built-in defaults, so nothing is tuned to your app yet. Generate a config for your use case with the MACA plugin, or write one by hand. The steps are below.")))
    : null;
  show(app, head, starter,
    h("div", { class: "stats two" },
      h("div", { class: "stat" }, h("span", { class: "stat-title" }, "Status"),
        h("b", { class: c.valid === false ? "bad" : c.valid ? "good" : "" }, state),
        h("span", { class: "muted" }, c.valid ? "The server is using this file." : c.valid === false ? "The server keeps the last good config." : "The file is too large to check here.")),
      h("div", { class: "stat" }, h("span", { class: "stat-title" }, "File"),
        h("b", { class: "small" }, c.path), h("span", { class: "muted" }, "Changed " + when(c.modified_at) + (c.size_bytes != null ? ", " + num(c.size_bytes) + " bytes" : "")))),
    c.valid === false
      ? h("div", { class: "problems" }, icon("alert"), h("div", null, h("b", null, "The server cannot use this file as it is"), h("pre", null, c.problems.join("\n"))))
      : null,
    c.sections.length
      ? [h("h2", null, "Settings by section"),
        h("div", { class: "sections" }, c.sections.map((s) => h("div", { class: "section" }, h("b", null, s.name), h("span", { class: "muted" }, num(s.settings) + (s.settings === 1 ? " setting" : " settings")))))]
      : null,
    h("h2", null, "memory.yml"),
    c.truncated ? h("div", { class: "notice" }, "The file is large. Only the first part is shown.") : null,
    sourceView(c.text),
    c.starter
      ? [h("h2", null, "Make it yours"), setupSteps(c, true)]
      : h("details", { class: "howto" }, h("summary", null, "How to change it"), setupSteps(c, true)));
}

// ---------------------------------------------------------------- shell

function askForToken(message) {
  const input = h("input", { type: "password", placeholder: "SYNAP_API_TOKEN", autocomplete: "off", "aria-label": "API token" });
  show(app, h("form", {
    class: "token",
    onsubmit: (ev) => { ev.preventDefault(); setToken(input.value.trim()); route(); },
  },
  h("h1", null, "This server needs its token"),
  h("p", { class: "muted" }, message || "SYNAP_API_TOKEN is set on this server. Paste it to continue. It stays in this tab only."),
  input,
  h("button", { type: "submit", class: "primary" }, "Continue")));
  input.focus();
}

function highlight(section) {
  for (const a of document.querySelectorAll("[data-nav]")) a.classList.toggle("on", a.dataset.nav === section);
  document.getElementById("where").textContent = { overview: "Overview", memories: "Memories", requests: "Requests", entities: "Entities", config: "Config" }[section] || "Memory browser";
}

async function route() {
  const rid = ++renderId;
  const { parts, query } = parseHash();
  const section = parts[0] || "overview";
  highlight(section);
  try {
    overview = await api("/overview");
    if (rid !== renderId) return;
    modeBadge.hidden = !overview.scoping_mode;
    modeBadge.textContent = overview.scoping_mode || "";
    document.getElementById("instance").textContent = overview.instance_id || "";
    if (section === "overview") await overviewPage(query, rid);
    else if (section === "requests" && parts[1]) await requestDetailPage(parts[1], rid);
    else if (section === "requests") await requestsPage(query, rid);
    else if (section === "memories" && parts[1]) await memoryDetailPage(parts[1], rid);
    else if (section === "memories") await memoriesPage(query, rid);
    else if (section === "entities" && parts[1] && parts[2]) await entityDetailPage(parts[1], parts.slice(2).join("/"), rid);
    else if (section === "entities") await entitiesPage(parts[1], query, rid);
    else if (section === "config") await configPage(rid);
    else show(app, h("div", { class: "error" }, "No such page."));
  } catch (err) {
    if (rid !== renderId) return;
    if (err instanceof AuthNeeded) {
      askForToken(getToken() ? "That token was not accepted. Check SYNAP_API_TOKEN and try again." : undefined);
    } else if (err.status === 404) {
      show(app, h("div", { class: "error" }, err.message || "Not found."), back("Back to memories", "#/memories"));
    } else {
      show(app, h("div", { class: "error" }, "Could not load this page: " + err.message));
    }
  }
}

window.addEventListener("hashchange", route);
route();
