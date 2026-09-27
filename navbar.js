/* ═══════════════════════════════════════════════════════════
   IFelx Web — navbar
   Builds the top bar, the project quick-search and the mobile
   menu. Also exposes window.IFX (icons + project list) for
   footer.js and site.js, which load after this file.
   ══════════════════════════════════════════════════════════ */
(function () {
"use strict";

/* ── Icons ─────────────────────────────────────────────── */
const svg = (d, extra) =>
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"${extra || ""}>${d}</svg>`;

const ICON = {
    search:   svg('<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>'),
    close:    svg('<path d="M18 6 6 18M6 6l12 12"/>'),
    menu:     svg('<path d="M4 7h16M4 12h16M4 17h16"/>'),
    external: svg('<path d="M7 17 17 7M8 7h9v9"/>'),
    arrow:    svg('<path d="M5 12h14M13 6l6 6-6 6"/>'),
    back:     svg('<path d="M19 12H5M11 18l-6-6 6-6"/>'),
    up:       svg('<path d="M12 19V5M6 11l6-6 6 6"/>'),
    download: svg('<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>'),
    copy:     svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/>'),
    check:    svg('<path d="m5 12 5 5L20 7"/>'),
    prev:     svg('<path d="m15 18-6-6 6-6"/>'),
    next:     svg('<path d="m9 18 6-6-6-6"/>'),
    search2:  svg('<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5M8 11h6M11 8v6"/>'),
    folder:   svg('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'),
    instagram: svg('<rect x="3" y="3" width="18" height="18" rx="5"/><circle cx="12" cy="12" r="4"/><circle cx="17.5" cy="6.5" r="0.6" fill="currentColor"/>'),
    github:   svg('<path d="M9 19c-4.3 1.4-4.3-2.5-6-3m12 5v-3.5c0-1 .1-1.4-.5-2 2.8-.3 5.5-1.4 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.2 4.2 0 0 0-.1-3.2s-1.1-.3-3.5 1.3a12.3 12.3 0 0 0-6.2 0C6.5 2.8 5.4 3.1 5.4 3.1a4.2 4.2 0 0 0-.1 3.2A4.6 4.6 0 0 0 4 9.5c0 4.6 2.7 5.7 5.5 6-.6.6-.6 1.2-.5 2V21"/>'),
};

/* ── Projects (single source for quick-search) ─────────── */
const ENGINE_URL = "https://ifelx.tailce0b52.ts.net";

const PROJECTS = [
    { name: "Jowa-mAi",         desc: "Compact open-source MiniGPT",           href: "/jowa-gpt/index.html",         icon: "/jowa.png",  keys: "jowa gpt mai ai llm minigpt lstm pytorch chat" },
    { name: "Jowa Football AI", desc: "Football game with self-learning AI",   href: "/jowa-football-ai/index.html", icon: "/jowa.png",  keys: "football soccer game rl dqn reinforcement" },
    { name: "IFelxOS",          desc: "Debian-based Linux distribution",       href: "/ifelx/index.html",            icon: "/ifelx.png", keys: "ifelxos os linux debian distro xv4 iso cinnamon" },
    { name: "IFelx Engine",     desc: "Independent search engine",             href: ENGINE_URL,                     icon: "/logo.png",  keys: "ifelx engine search wex crawler index ifse" },
    { name: "OpenWebfilemgr",   desc: "Open-source web file manager",          href: "https://github.com/AhmedXV-V12/OpenWebfilemgr", icon: "/logo.png", keys: "file manager web github open" },
    { name: "xv4 Package Manager", desc: "IFelx Shop — packages for IFelxOS", href: "/IFelx-shop/xv4/index.html",   icon: "/ifelx.png", keys: "xv4 shop package manager install update upgrade ifos" },
];

window.IFX = { icon: ICON, projects: PROJECTS, engineUrl: ENGINE_URL };

const LINKS = [
    { label: "Jowa-mAi",     href: "/jowa-gpt/index.html" },
    { label: "Football AI",  href: "/jowa-football-ai/index.html" },
    { label: "IFelxOS",      href: "/ifelx/index.html" },
    { label: "IFelx Engine", href: ENGINE_URL },
];

const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isExternal = href => /^https?:\/\//.test(href);
const ext = href => isExternal(href) ? ' target="_blank" rel="noopener"' : "";
const section = path => (path.split("/").filter(Boolean)[0] || "").toLowerCase();

/* ── Markup ────────────────────────────────────────────── */
const here = section(location.pathname);

const linksHtml = LINKS.map(l => {
    const current = !isExternal(l.href) && here && section(l.href) === here;
    return `<a href="${l.href}"${ext(l.href)}${current ? ' class="is-current" aria-current="page"' : ""}>${esc(l.label)}</a>`;
}).join("");

const navbar = document.createElement("nav");
navbar.className = "navbar";
navbar.setAttribute("aria-label", "Main");
navbar.innerHTML = `
    <div class="navbar-inner">
        <a href="/index.html" class="navbar-brand" aria-label="IFelx Web — home">
            <img src="/logo.png" alt="" width="36" height="36">
            <span class="navbar-brand-name">IFelx Web</span>
        </a>
        <div class="navbar-panel" id="navPanel">
            <div class="navbar-search" role="search">
                <div class="navbar-search-box">
                    ${ICON.search}
                    <input type="search" id="navSearch" placeholder="Search projects..." autocomplete="off" spellcheck="false"
                           role="combobox" aria-expanded="false" aria-controls="navResults" aria-autocomplete="list" aria-label="Search projects">
                    <kbd class="navbar-kbd" aria-hidden="true">/</kbd>
                    <button type="button" class="navbar-search-clear" aria-label="Clear search">${ICON.close}</button>
                </div>
                <div class="navbar-results" id="navResults" role="listbox" aria-label="Projects"></div>
            </div>
            <div class="navbar-links">
                ${linksHtml}
                <a href="https://www.instagram.com/axv.ifelx" target="_blank" rel="noopener" class="navbar-instagram" aria-label="Instagram @axv.ifelx">
                    <span class="navbar-insta-icon">${ICON.instagram}</span>
                    axv.ifelx
                </a>
            </div>
        </div>
        <button type="button" class="navbar-menu-btn" aria-label="Menu" aria-expanded="false" aria-controls="navPanel">
            <span class="icon-open">${ICON.menu}</span>
            <span class="icon-close">${ICON.close}</span>
        </button>
    </div>
`;
document.body.insertBefore(navbar, document.body.firstChild);

/* ── Scroll shadow ─────────────────────────────────────── */
const onScroll = () => navbar.classList.toggle("is-scrolled", window.scrollY > 8);
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

/* ── Mobile menu ───────────────────────────────────────── */
const menuBtn = navbar.querySelector(".navbar-menu-btn");
const setMenu = open => {
    navbar.classList.toggle("is-menu-open", open);
    menuBtn.setAttribute("aria-expanded", String(open));
};
menuBtn.addEventListener("click", () => setMenu(!navbar.classList.contains("is-menu-open")));
document.addEventListener("click", e => { if (!navbar.contains(e.target)) setMenu(false); });
window.addEventListener("resize", () => { if (window.innerWidth > 767) setMenu(false); });

/* ── Search ────────────────────────────────────────────── */
const wrap    = navbar.querySelector(".navbar-search");
const input   = navbar.querySelector("#navSearch");
const clear   = navbar.querySelector(".navbar-search-clear");
const results = navbar.querySelector("#navResults");
let matches = [];
let active  = -1;

const norm = s => s.toLowerCase().trim();

const highlight = (text, q) => {
    const i = q ? text.toLowerCase().indexOf(q) : -1;
    if (i < 0) return esc(text);
    return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
};

/* index page: filter the projects grid and status cards in place */
const filterPage = q => {
    const grid = document.getElementById("projectsGrid");
    if (!grid) return;
    let shown = 0;
    document.querySelectorAll("#projectsGrid .project-btn, .status-section .status-card").forEach(el => {
        const hay = norm((el.getAttribute("data-name") || "") + " " + el.textContent);
        const ok = !q || hay.includes(q);
        el.hidden = !ok;
        if (ok && el.classList.contains("project-btn")) shown++;
    });
    const empty = document.getElementById("searchEmpty");
    if (empty) {
        empty.classList.toggle("show", !!q && shown === 0);
        const term = empty.querySelector("strong");
        if (term) term.textContent = input.value.trim();
    }
};

const setActive = i => {
    const items = results.querySelectorAll(".navbar-result");
    items.forEach((el, n) => el.classList.toggle("is-active", n === i));
    active = i;
    if (items[i]) {
        items[i].scrollIntoView({ block: "nearest" });
        input.setAttribute("aria-activedescendant", items[i].id);
    } else {
        input.removeAttribute("aria-activedescendant");
    }
};

const open = show => {
    wrap.classList.toggle("is-open", show);
    input.setAttribute("aria-expanded", String(show));
};

const render = () => {
    const q = norm(input.value);
    wrap.classList.toggle("has-value", !!input.value);
    filterPage(q);

    matches = PROJECTS.filter(p => !q || norm(p.name + " " + p.desc + " " + p.keys).includes(q));

    if (!matches.length) {
        results.innerHTML = `<div class="navbar-results-empty">No projects match “${esc(input.value.trim())}”</div>`;
    } else {
        results.innerHTML = matches.map((p, i) => `
            <a class="navbar-result" id="navResult${i}" role="option" href="${p.href}"${ext(p.href)}>
                <img src="${p.icon}" alt="" width="34" height="34" loading="lazy">
                <span class="navbar-result-text">
                    <span class="navbar-result-name">${highlight(p.name, q)}</span>
                    <span class="navbar-result-desc">${esc(p.desc)}</span>
                </span>
                ${isExternal(p.href) ? ICON.external : ICON.arrow}
            </a>`).join("");
    }
    active = -1;
    input.removeAttribute("aria-activedescendant");
    open(document.activeElement === input);
};

input.addEventListener("input", render);
input.addEventListener("focus", render);

input.addEventListener("keydown", e => {
    const n = matches.length;
    if (e.key === "ArrowDown" && n) { e.preventDefault(); open(true); setActive((active + 1) % n); }
    else if (e.key === "ArrowUp" && n) { e.preventDefault(); setActive((active - 1 + n) % n); }
    else if (e.key === "Enter" && n) {
        e.preventDefault();
        const item = results.querySelectorAll(".navbar-result")[active < 0 ? 0 : active];
        if (item) item.click();
    }
    else if (e.key === "Escape") {
        if (input.value) { input.value = ""; render(); }
        else { open(false); input.blur(); setMenu(false); }
    }
});

clear.addEventListener("click", () => { input.value = ""; render(); input.focus(); });

document.addEventListener("click", e => { if (!wrap.contains(e.target)) open(false); });

/* "/" focuses search from anywhere (opens the mobile menu if needed) */
document.addEventListener("keydown", e => {
    if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    e.preventDefault();
    if (window.innerWidth <= 767) setMenu(true);
    input.focus();
});
})();
