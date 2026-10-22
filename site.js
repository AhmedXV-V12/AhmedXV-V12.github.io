/* ═══════════════════════════════════════════════════════════
   IFelx Web — shared page behaviours
   Copy buttons on code blocks, screenshot lightbox, scroll
   reveal, external-link handling. Loaded after navbar.js.
   ══════════════════════════════════════════════════════════ */
(function () {
"use strict";

const ICON = (window.IFX && window.IFX.icon) || {};
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ── External links open in a new tab ──────────────────── */
document.querySelectorAll('a[href^="http"]').forEach(a => {
    if (a.host === location.host) return;
    a.target = "_blank";
    a.rel = "noopener";
});

/* ── Button icons ──────────────────────────────────────── */
document.querySelectorAll(".download-buttons .project-btn").forEach(btn => {
    if (!btn.querySelector(".btn-icon") && ICON.download) {
        btn.insertAdjacentHTML("afterbegin", ICON.download.replace("<svg", '<svg class="btn-icon"'));
    }
});

document.querySelectorAll(".back-link").forEach(a => {
    a.textContent = a.textContent.replace(/^\s*←\s*/, "");
    if (ICON.back) a.insertAdjacentHTML("afterbegin", ICON.back.replace("<svg", '<svg width="14" height="14"'));
});

/* ── Toast ─────────────────────────────────────────────── */
let toastEl, toastTimer;
const toast = msg => {
    if (!toastEl) {
        toastEl = document.createElement("div");
        toastEl.className = "ifx-toast";
        toastEl.setAttribute("role", "status");
        document.body.appendChild(toastEl);
    }
    toastEl.innerHTML = (ICON.check || "") + "<span></span>";
    toastEl.querySelector("span").textContent = msg;
    requestAnimationFrame(() => toastEl.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1800);
};

/* ── Copy buttons on code blocks ───────────────────────── */
const copyText = async text => {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (_) {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        let ok = false;
        try { ok = document.execCommand("copy"); } catch (_) {}
        ta.remove();
        return ok;
    }
};

/* text is read at click time so pages can rewrite a block's command */
const blockText = block => {
    const clone = block.cloneNode(true);
    clone.querySelectorAll(".code-copy").forEach(el => el.remove());
    return clone.textContent.replace(/^\n+/, "").replace(/\s+$/, "");
};

document.querySelectorAll(".code-block").forEach(block => {
    // markup starts code on the line after <div>; drop that empty first line
    const first = block.firstChild;
    if (first && first.nodeType === 3) first.textContent = first.textContent.replace(/^\s*\n/, "");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-copy";
    btn.setAttribute("aria-label", "Copy code");
    btn.innerHTML = ICON.copy || "Copy";
    btn.addEventListener("click", async () => {
        if (!(await copyText(blockText(block)))) return;
        btn.classList.add("is-done");
        btn.innerHTML = ICON.check || "✓";
        toast("Copied to clipboard");
        setTimeout(() => { btn.classList.remove("is-done"); btn.innerHTML = ICON.copy || "Copy"; }, 1600);
    });
    block.classList.add("has-copy");
    block.appendChild(btn);
});

/* ── Screenshot lightbox ───────────────────────────────── */
const shots = Array.from(document.querySelectorAll(".screenshots-grid img"));
if (shots.length) {
    const box = document.createElement("div");
    box.className = "ifx-lightbox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Screenshot viewer");
    box.innerHTML = `
        <img alt="">
        <button type="button" class="ifx-lightbox-btn close" aria-label="Close">${ICON.close || "×"}</button>
        <button type="button" class="ifx-lightbox-btn prev" aria-label="Previous">${ICON.prev || "‹"}</button>
        <button type="button" class="ifx-lightbox-btn next" aria-label="Next">${ICON.next || "›"}</button>
        <div class="ifx-lightbox-caption"></div>`;
    document.body.appendChild(box);

    const img = box.querySelector("img");
    const cap = box.querySelector(".ifx-lightbox-caption");
    let index = 0, lastFocus = null;

    const show = i => {
        index = (i + shots.length) % shots.length;
        img.src = shots[index].currentSrc || shots[index].src;
        img.alt = shots[index].alt;
        cap.textContent = `${shots[index].alt || "Screenshot"} · ${index + 1} / ${shots.length}`;
    };
    const openBox = i => {
        lastFocus = document.activeElement;
        show(i);
        box.classList.add("is-open");
        document.documentElement.style.overflow = "hidden";
        box.querySelector(".close").focus();
    };
    const closeBox = () => {
        box.classList.remove("is-open");
        document.documentElement.style.overflow = "";
        if (lastFocus) lastFocus.focus();
    };

    shots.forEach((s, i) => {
        s.tabIndex = 0;
        s.setAttribute("role", "button");
        s.loading = "lazy";
        s.addEventListener("click", () => openBox(i));
        s.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openBox(i); } });
    });

    box.querySelector(".close").addEventListener("click", closeBox);
    box.querySelector(".prev").addEventListener("click", () => show(index - 1));
    box.querySelector(".next").addEventListener("click", () => show(index + 1));
    box.addEventListener("click", e => { if (e.target === box) closeBox(); });

    document.addEventListener("keydown", e => {
        if (!box.classList.contains("is-open")) return;
        if (e.key === "Escape") closeBox();
        else if (e.key === "ArrowLeft") show(index - 1);
        else if (e.key === "ArrowRight") show(index + 1);
        else if (e.key === "Tab") {
            // keep focus on the viewer's buttons while it is open (it is modal)
            const btns = Array.from(box.querySelectorAll("button"));
            const at = btns.indexOf(document.activeElement);
            e.preventDefault();
            btns[(at + (e.shiftKey ? -1 : 1) + btns.length) % btns.length].focus();
        }
    });

    let touchX = null;
    box.addEventListener("touchstart", e => { touchX = e.touches[0].clientX; }, { passive: true });
    box.addEventListener("touchend", e => {
        if (touchX === null) return;
        const dx = e.changedTouches[0].clientX - touchX;
        if (Math.abs(dx) > 50) show(index + (dx < 0 ? 1 : -1));
        touchX = null;
    });
}

/* ── Scroll reveal for content cards ───────────────────── */
if (!reduceMotion && "IntersectionObserver" in window) {
    const targets = document.querySelectorAll(".page-container > .card, .page-container > .version-card, .page-container > .download-buttons, .page-container > .legacy-notice");
    const io = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add("is-in");
            io.unobserve(entry.target);
        });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });

    // measured in the first frame, when layout is done anyway (no forced reflow at load)
    requestAnimationFrame(() => targets.forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.top < window.innerHeight) return;   // already on screen: no flash
        el.classList.add("ifx-reveal");
        io.observe(el);
    }));
}

/* ── "On this page": contents for long pages ───────────── */
/* Built from the section headings; shown beside the text on wide screens
   (CSS), with the section being read marked as you scroll. */
const container = document.querySelector(".page-container");
const tocSections = container ? Array.from(container.querySelectorAll(":scope > .card")).filter(s => s.querySelector(":scope > h2")) : [];
if (tocSections.length >= 3) {
    const slug = t => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    const nav = document.createElement("nav");
    nav.className = "page-toc";
    nav.setAttribute("aria-label", "On this page");
    nav.innerHTML = '<p class="page-toc-title">On this page</p><ol></ol>';
    const list = nav.querySelector("ol");
    const links = new Map();
    tocSections.forEach((sec, i) => {
        const h = sec.querySelector(":scope > h2");
        if (!sec.id) sec.id = slug(h.textContent) || "section-" + (i + 1);
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "#" + sec.id;
        // the heading text without badges or icons inside it
        a.textContent = Array.from(h.childNodes).filter(n => n.nodeType === 3 || n.tagName === "CODE").map(n => n.textContent).join("").trim();
        li.appendChild(a);
        list.appendChild(li);
        links.set(sec, a);
    });
    const header = container.querySelector(":scope > .page-header");
    if (header) header.after(nav); else container.prepend(nav);
    // the contents column starts on the row after the header (a back link may come first)
    const kids = Array.from(container.children);
    nav.style.setProperty("--toc-row", String(header ? kids.indexOf(header) + 2 : 1));
    container.classList.add("has-toc");

    if ("IntersectionObserver" in window) {
        const visible = new Set();
        const mark = () => {
            const first = tocSections.find(s => visible.has(s));
            links.forEach((a, s) => {
                const on = s === first;
                a.classList.toggle("is-current", on);
                if (on) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
            });
        };
        const spy = new IntersectionObserver(entries => {
            entries.forEach(e => e.isIntersecting ? visible.add(e.target) : visible.delete(e.target));
            mark();
        }, { rootMargin: "-20% 0px -55% 0px" });
        tocSections.forEach(s => spy.observe(s));
    }
}

/* ── Version badge + release history ───────────────────── */
/* Official release history of IFelx Web, newest first; the first entry is the
   version shown in the corner. Every update is one release and versions count up
   one step at a time: 1.0 ... 1.9, 2.0 ... 2.9, 3.0 ... (never 1.10). */
const RELEASES = [
    { v: "7.1", date: "2026-10-22", changes: [
        "IFOS subsystem page: itelxOS 1.4, with its new sys commands (sys, sys.info, sys.info.device, sys.all and sys.top) and screenshots of them; the download is ITelxOS 1.4.",
        "How to update itelxOS 1.3 to 1.4 with update.xvr from a USB stick, step by step, with the update and a stick image that starts by itself to download.",
        "itelxOS 1.4 in the version history; 1.3 and the first version, 1.2, stay downloadable.",
        "IFelxOS2.5 on the IFOS subsystem page: the interface update of IFelxOS2, with screenshots, a download and a side-by-side comparison with IFelxOS2, which stays downloadable.",
        "The home page, the project index and the quick search name itelxOS 1.4 and IFelxOS2.5.",
    ] },
    { v: "7.0", date: "2026-10-16", changes: [
        "Redesigned home page: a new hero with the IFelx mark, featured work with real screens from IFOS, Jowa Football AI v3 and IFelxOS, and one grouped index of every project with its status.",
        "New navigation: a Projects menu with every project and its logo, direct links to IFelx Engine and IFelxChat, and a full-screen menu on phones.",
        "Project pages share one layout: a large title beside the project logo, sections in one reading column, and an \"On this page\" list on wide screens that follows your scrolling.",
        "Faster pages: the Inter font is now served by the site itself instead of Google Fonts, and logos and screenshots come in small WebP copies.",
        "Easier to read: grey text is brighter so every label meets the WCAG AA contrast ratio.",
        "Keyboard: a \"Skip to content\" link on every page, and Escape closes the mobile menu.",
        "Search engines and link previews: every page has its own description, canonical address and preview card.",
        "The IFOS subsystem card and quick search name itelxOS 1.3, and the footer links to the IFOS subsystem page.",
    ] },
    { v: "6.9", date: "2026-10-11", changes: [
        "IFOS subsystem page: itelxOS 1.3, with USB, FAT storage and programs from USB sticks; the download is ITelxOS 1.3.",
        "A version history of itelxOS on the same page, with what each version brought and a download for every version, the first one (1.2) too.",
    ] },
    { v: "6.8", date: "2026-10-10", changes: [
        "Every update of the site is now its own release, including the small ones.",
        "Versions count up one step per update: 1.0 to 1.9, then 2.0, and so on.",
    ] },
    { v: "6.7", date: "2026-10-10", changes: [
        "Release history extended back to the first version of the site, written from its full GitHub history.",
    ] },
    { v: "6.6", date: "2026-10-10", changes: [
        "Version number in the bottom corner of every page; selecting it opens this release history.",
    ] },
    { v: "6.5", date: "2026-10-10", changes: [
        "Jowa Football AI v3 page: 11v11 football played by two AI brains (MAPPO with an entity transformer), with specs and pictures from real matches.",
        "Downloads for v3: Windows and Linux apps with the default weights, and the source code.",
        "v3 is now the latest version on the Jowa Football AI page; the quick search, footer and home page point to it.",
    ] },
    { v: "6.4", date: "2026-10-09", changes: [
        "IFelxChat added: button and status card on the home page, links in the navbar and the footer.",
    ] },
    { v: "6.3", date: "2026-10-08", changes: [
        "New IFOS subsystem page: IFOS2, IFelxOS2, the ODE installer and itelxOS 1.2, with screenshots and downloads.",
        "IFOS subsystem in the navbar, marked New.",
        "IFelx Shop pages (xv4, install, update, upgrade) moved to the site design.",
    ] },
    { v: "6.2", date: "2026-09-27", changes: [
        "New design for the whole site in the IFelx identity: dark theme, blue accent, rounded cards and pill buttons.",
        "Shared navbar with a project quick search (press /) and a phone menu, and a shared footer on every page.",
        "Copy buttons on code blocks, a screenshot viewer, and cards that fade in while scrolling.",
    ] },
    { v: "6.1", date: "2026-09-26", changes: [
        "IFelxOS 1 download moved to the axv-1 server.",
    ] },
    { v: "6.0", date: "2026-09-25", changes: [
        "IFelx Engine status card: new name and the IFelx icon.",
    ] },
    { v: "5.9", date: "2026-09-03", changes: [
        "WEX Engine renamed to IFelx Engine on the home page.",
    ] },
    { v: "5.8", date: "2026-08-16", changes: [
        "Jowa Football AI status card: \"Artificial intelligence plays 6-a-side football\".",
    ] },
    { v: "5.7", date: "2026-08-07", changes: [
        "OpenWebfilemgr added to the home page.",
        "WEX Engine described as an independent search engine with its own crawling and indexing.",
    ] },
    { v: "5.6", date: "2026-07-30", changes: [
        "xv4.sh package removed from the xv4t1 channel and its package index.",
    ] },
    { v: "5.5", date: "2026-07-20", changes: [
        "New home page: \"Official platform of AXV\" with Open Source, AI Powered and Active badges, project buttons and a Project Status section.",
        "Project icons for Jowa-mAi, Jowa Football AI, IFelxOS and WEX Engine; page title \"IFelx Web — AXV Official\".",
    ] },
    { v: "5.4", date: "2026-07-10", changes: [
        "WEX Engine button available again.",
    ] },
    { v: "5.3", date: "2026-07-09", changes: [
        "packages.json package index for IFelx Shop and the xv4t1 channel.",
        "New build of the xv4.sh package.",
    ] },
    { v: "5.2", date: "2026-07-05", changes: [
        "Shorter footer text, signed AXV.",
    ] },
    { v: "5.1", date: "2026-07-03", changes: [
        "WEX Engine marked \"not available now\" and its button disabled.",
    ] },
    { v: "5.0", date: "2026-06-20", changes: [
        "systemOS update script (b.sh): typo fixed.",
    ] },
    { v: "4.9", date: "2026-05-24", changes: [
        "Footer: WEX Engine links point to the main address of the engine.",
    ] },
    { v: "4.8", date: "2026-05-23", changes: [
        "Shared footer on every page: projects, downloads, Instagram and GitHub.",
        "New logo, and pictures back on the Jowa Football AI pages.",
    ] },
    { v: "4.7", date: "2026-05-21", changes: [
        "systemOS update script (b.sh) updated.",
    ] },
    { v: "4.6", date: "2026-05-20", changes: [
        "Rebuilt navbar with a project search box and the Instagram account axv.ifelx.",
        "Home page back: WEX Engine, the search engine at ifelx.tailce0b52.ts.net, replaces WEX Browser.",
        "IFelxOS 1 is downloaded from the IFelx server instead of Google Drive.",
        "New logo; sitemap, robots.txt and Google Search verification updated.",
    ] },
    { v: "4.5", date: "2026-05-19", changes: [
        "Home page and navbar taken down for the rebuild.",
    ] },
    { v: "4.4", date: "2026-05-17", changes: [
        "Google Search verification file updated.",
    ] },
    { v: "4.3", date: "2026-04-21", changes: [
        "IFelxOS 1 download moved to a new Google Drive link.",
        "New Google Search verification.",
    ] },
    { v: "4.2", date: "2026-03-17", changes: [
        "IFelx_mobile folder removed.",
    ] },
    { v: "4.1", date: "2026-03-13", changes: [
        "Instagram link changed from axv.bin to axv.ifelx.",
    ] },
    { v: "4.0", date: "2026-03-11", changes: [
        "IFelx_mobile folder created.",
    ] },
    { v: "3.9", date: "2026-03-09", changes: [
        "Jowa Football AI versions page: v1 (3v3, Q-table, Python script) and v2 (6v6, DQN, ZIP package), each with its own page.",
    ] },
    { v: "3.8", date: "2026-03-08", changes: [
        "Shared navbar on every page, with an Instagram link (axv.bin).",
        "JOWA-GPT becomes Jowa-mAi: new page with v1.6 hosted on the site, GNU GPL v2.0 licence and an explanation of the LSTM model.",
        "IFelxOS page rebuilt with a specs table; home buttons renamed.",
    ] },
    { v: "3.7", date: "2026-03-07", changes: [
        "First shared navbar: IFelx Web, Jowa-GPT, Football AI, IFelxOS and WEX.",
        "Project list removed from the home page.",
    ] },
    { v: "3.6", date: "2026-02-08", changes: [
        "Home server survey page removed.",
    ] },
    { v: "3.5", date: "2026-02-07", changes: [
        "Background images for the site.",
        "IFelxOS page restyled, with a direct download link.",
        "Survey page about a home server (homeserver).",
    ] },
    { v: "3.4", date: "2026-02-06", changes: [
        "New logo.",
        "\"What is IFelx Web?\" introduction on the home page.",
    ] },
    { v: "3.3", date: "2026-02-05", changes: [
        "First shared stylesheet (style.css): new fonts and project buttons on the home page.",
    ] },
    { v: "3.2", date: "2026-02-04", changes: [
        "Jowa Football AI download: the script no longer needs PyTorch.",
    ] },
    { v: "3.1", date: "2026-01-31", changes: [
        "IFelxOS page: IFelxOS 1, a Debian-based distribution with Cinnamon, apt and xv4.sh (4.5 GB ISO, released 1 Feb 2026), its goals, screenshots and download.",
        "\"New: Install IFelxOS\" button on the home page.",
    ] },
    { v: "3.0", date: "2026-01-24", changes: [
        "New home page with the list of AXV projects: jowa-GPT, jowa-football-ai, jowa-e, jowa-ifelx, xv2-xv5, IFelxOS, IFelx-mini, IFelxUI and WEX browser.",
        "\"New: WEX browser\" link and a new logo.",
    ] },
    { v: "2.9", date: "2026-01-16", changes: [
        "New description of the site for search engines.",
    ] },
    { v: "2.8", date: "2026-01-15", changes: [
        "Home page: IFelxOS release date announced, author shown as AXV.",
        "systemOS update script (b.sh).",
        "New build of the xv4.sh package.",
    ] },
    { v: "2.7", date: "2026-01-12", changes: [
        "New build of the xv4.sh package.",
    ] },
    { v: "2.6", date: "2026-01-10", changes: [
        "New build of the xv4.sh package.",
    ] },
    { v: "2.5", date: "2026-01-08", changes: [
        "axv, joda and xv3 packages in the xv4t1 channel.",
    ] },
    { v: "2.4", date: "2026-01-05", changes: [
        "xv4t1 package channel, starting with the xv4.sh package.",
    ] },
    { v: "2.3", date: "2026-01-04", changes: [
        "IFelx Shop: bat, busybox, fd, jq and ncdu packages.",
    ] },
    { v: "2.2", date: "2026-01-03", changes: [
        "IFelx Shop: 14 new packages, among them calculator_gui, notes_cli, password_safe, system_monitor and todo_cli.",
        "Update list rewritten as a list of package files.",
    ] },
    { v: "2.1", date: "2025-12-14", changes: [
        "IFelx Shop: apt_pkg_mgr 1.0 package.",
    ] },
    { v: "2.0", date: "2025-12-05", changes: [
        "Home page language set to English.",
    ] },
    { v: "1.9", date: "2025-12-04", changes: [
        "The site takes its name, IFelx Web (was IFlex Web).",
    ] },
    { v: "1.8", date: "2025-12-03", changes: [
        "Bing Webmaster verification.",
        "arabicxv link spelled correctly on the home page.",
    ] },
    { v: "1.7", date: "2025-11-30", changes: [
        "IFelx Shop: cute.sh package, which deletes invalid .desktop files.",
        "Windows commands removed from the xv4 update page.",
    ] },
    { v: "1.6", date: "2025-11-29", changes: [
        "IFelx Shop update list: note for the jotxt package.",
    ] },
    { v: "1.5", date: "2025-11-28", changes: [
        "IFelx Shop: pages for the xv4 package manager (install, update, upgrade), an update list and the first package, jotxt.",
        "Site renamed from IFex Web to IFlex Web.",
        "Google Search verification.",
    ] },
    { v: "1.4", date: "2025-11-27", changes: [
        "Home page introduction; the arabicxv link becomes \"arabicx Topic Search Engine\".",
    ] },
    { v: "1.3", date: "2025-11-26", changes: [
        "arabicxv: an Arabic topic search page, linked from the home page.",
        "Logo and copyright line on the home page.",
    ] },
    { v: "1.2", date: "2025-11-23", changes: [
        "Site icon (favicon).",
    ] },
    { v: "1.1", date: "2025-11-22", changes: [
        "\"Official Site\" title, robots.txt and sitemap.xml for search engines.",
    ] },
    { v: "1.0", date: "2025-11-21", changes: [
        "IFex Web goes online: home page with Jowa-GPT and Jowa Football AI; IFelxOS and IFelx Shop announced as coming soon.",
        "JOWA-GPT page: JOWA-MiniGPT version 1, a compact language model with Arabic data, downloaded from Google Drive.",
        "Jowa Football AI page: a 3v3 football game with Q-table AI for 1 to 3 players, download for Windows and Linux.",
    ] },
];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const fmtDate = iso => {
    const [y, m, d] = iso.split("-").map(Number);
    return `${d} ${MONTHS[m - 1]} ${y}`;
};

const versionBtn = document.createElement("button");
versionBtn.type = "button";
versionBtn.className = "ifx-version";
versionBtn.textContent = "v" + RELEASES[0].v;
// the accessible name starts with the visible text (WCAG 2.5.3, label in name)
versionBtn.setAttribute("aria-label", `v${RELEASES[0].v}: IFelx Web release history`);
versionBtn.setAttribute("aria-haspopup", "dialog");
document.body.appendChild(versionBtn);

let changelog = null;
const buildChangelog = () => {
    const dlg = document.createElement("dialog");
    dlg.className = "ifx-changelog";
    dlg.setAttribute("aria-labelledby", "ifx-changelog-title");
    dlg.innerHTML = `
        <div class="ifx-changelog-panel">
            <div class="ifx-changelog-head">
                <div>
                    <span class="ifx-changelog-kicker">IFelx Web · v${RELEASES[0].v}</span>
                    <h2 id="ifx-changelog-title">Release history</h2>
                    <p class="ifx-changelog-sub">${RELEASES.length} releases since ${fmtDate(RELEASES[RELEASES.length - 1].date)}</p>
                </div>
                <button type="button" class="ifx-changelog-close" aria-label="Close">${ICON.close || "×"}</button>
            </div>
            <ol class="ifx-changelog-list"></ol>
        </div>`;
    const list = dlg.querySelector(".ifx-changelog-list");
    const major = r => r.v.split(".")[0];
    let groupList = null;
    RELEASES.forEach((r, i) => {
        if (i === 0 || major(r) !== major(RELEASES[i - 1])) {
            // one group per major version, headed "Version 1.x · 21 Nov 2025 – 26 Sep 2026";
            // the sticky heading stays inside its own group, so the next one pushes it away
            const members = RELEASES.filter(x => major(x) === major(r));
            const group = document.createElement("li");
            group.className = "ifx-era";
            group.innerHTML = '<div class="ifx-release-era"></div><ol></ol>';
            group.firstChild.textContent = `Version ${major(r)}.x · ${fmtDate(members[members.length - 1].date)} – ${fmtDate(members[0].date)}`;
            list.appendChild(group);
            groupList = group.lastChild;
        }
        const li = document.createElement("li");
        li.className = "ifx-release";
        li.innerHTML = `
            <div class="ifx-release-head">
                <span class="ifx-release-v"></span>
                <time></time>
                ${i === 0 ? '<span class="ifx-release-latest">Latest</span>' : ""}
            </div>
            <ul></ul>`;
        li.querySelector(".ifx-release-v").textContent = r.v;
        const t = li.querySelector("time");
        t.dateTime = r.date;
        t.textContent = fmtDate(r.date);
        const ul = li.querySelector("ul");
        r.changes.forEach(c => {
            const item = document.createElement("li");
            item.textContent = c;
            ul.appendChild(item);
        });
        groupList.appendChild(li);
    });
    // "close" is also fired for Esc, but Chromium delays it until the next rendered frame,
    // so the buttons restore the page themselves; running twice is harmless
    const restore = () => {
        document.documentElement.style.overflow = "";
        versionBtn.focus();
    };
    const closeIt = () => { dlg.close(); restore(); };
    dlg.querySelector(".ifx-changelog-close").addEventListener("click", closeIt);
    dlg.addEventListener("click", e => { if (e.target === dlg) closeIt(); });   // backdrop
    dlg.addEventListener("close", restore);
    document.body.appendChild(dlg);
    return dlg;
};

const openChangelog = () => {
    changelog = changelog || buildChangelog();
    if (changelog.open) return;
    changelog.showModal();
    document.documentElement.style.overflow = "hidden";
    changelog.querySelector(".ifx-changelog-list").scrollTop = 0;
};

versionBtn.addEventListener("click", openChangelog);
if (location.hash === "#changelog") openChangelog();
})();
