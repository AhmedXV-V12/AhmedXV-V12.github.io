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

    targets.forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.top < window.innerHeight) return;   // already on screen: no flash
        el.classList.add("ifx-reveal");
        io.observe(el);
    });
}

/* ── Version badge + release history ───────────────────── */
/* Official release history of IFelx Web, newest first. Versions are two numbers
   (major.minor); the first entry is the version shown in the corner. */
const RELEASES = [
    { v: "2.4", date: "2026-10-10", changes: [
        "Version number in the bottom corner of every page; selecting it opens this release history.",
        "Official release history of IFelx Web back to the first version (1.0, November 2025), written from the site's full GitHub history.",
    ] },
    { v: "2.3", date: "2026-10-10", changes: [
        "Jowa Football AI v3 page: 11v11 football played by two AI brains (MAPPO with an entity transformer), with specs and pictures from real matches.",
        "Downloads for v3: Windows and Linux apps with the default weights, and the source code.",
        "v3 is now the latest version on the Jowa Football AI page; the quick search, footer and home page point to it.",
    ] },
    { v: "2.2", date: "2026-10-09", changes: [
        "IFelxChat added: button and status card on the home page, links in the navbar and the footer.",
    ] },
    { v: "2.1", date: "2026-10-08", changes: [
        "New IFOS subsystem page: IFOS2, IFelxOS2, the ODE installer and itelxOS 1.2, with screenshots and downloads.",
        "IFOS subsystem in the navbar, marked New.",
        "IFelx Shop pages (xv4, install, update, upgrade) moved to the site design.",
    ] },
    { v: "2.0", date: "2026-09-27", changes: [
        "New design for the whole site in the IFelx identity: dark theme, blue accent, rounded cards and pill buttons.",
        "Shared navbar with a project quick search (press /) and a phone menu, and a shared footer on every page.",
        "Copy buttons on code blocks, a screenshot viewer, and cards that fade in while scrolling.",
    ] },
    { v: "1.17", date: "2026-09-26", changes: [
        "WEX Engine renamed to IFelx Engine (3 Sep), with its new icon (25 Sep).",
        "IFelxOS 1 download moved to the axv-1 server.",
    ] },
    { v: "1.16", date: "2026-08-16", changes: [
        "OpenWebfilemgr added to the home page (7 Aug).",
        "Status cards: WEX Engine is described as an independent search engine with its own crawling and indexing, and Jowa Football AI as AI playing 6-a-side football.",
    ] },
    { v: "1.15", date: "2026-07-20", changes: [
        "New home page: \"Official platform of AXV\" with Open Source, AI Powered and Active badges, project buttons and a Project Status section.",
        "Project icons for Jowa-mAi, Jowa Football AI, IFelxOS and WEX Engine.",
        "xv4 package refreshed in the xv4t1 channel (30 Jul).",
    ] },
    { v: "1.14", date: "2026-07-10", changes: [
        "packages.json package index for IFelx Shop and the xv4t1 channel (9 Jul).",
        "WEX Engine was marked unavailable on 3 Jul and came back on 10 Jul.",
        "Shorter footer text (5 Jul).",
    ] },
    { v: "1.13", date: "2026-05-23", changes: [
        "Shared footer on every page: projects, downloads, Instagram and GitHub.",
        "New logo, and pictures back on the Jowa Football AI pages.",
    ] },
    { v: "1.12", date: "2026-05-20", changes: [
        "Rebuilt navbar with a project search box and the Instagram account axv.ifelx.",
        "WEX Engine, the search engine at ifelx.tailce0b52.ts.net, replaces WEX Browser on the home page.",
        "IFelxOS 1 is downloaded from the IFelx server instead of Google Drive (new Drive link on 21 Apr).",
        "Sitemap, robots.txt and Google Search verification updated.",
    ] },
    { v: "1.11", date: "2026-03-09", changes: [
        "Jowa Football AI versions page: v1 (3v3, Q-table, Python script) and v2 (6v6, DQN, ZIP package), each with its own page.",
        "Instagram link changed from axv.bin to axv.ifelx (13 Mar).",
    ] },
    { v: "1.10", date: "2026-03-08", changes: [
        "First shared navbar on every page: IFelx Web, Jowa-GPT, Football AI, IFelxOS, WEX and Instagram.",
        "JOWA-GPT becomes Jowa-mAi: new page with v1.6 hosted on the site, GNU GPL v2.0 licence and an explanation of the LSTM model.",
        "IFelxOS page rebuilt with a specs table.",
    ] },
    { v: "1.9", date: "2026-02-07", changes: [
        "First shared stylesheet (style.css) for the site (5 Feb), new logo and background images.",
        "\"What is IFelx Web?\" introduction on the home page.",
        "IFelxOS page restyled with a direct download link; Jowa Football AI download updated (4 Feb).",
    ] },
    { v: "1.8", date: "2026-01-31", changes: [
        "IFelxOS page: IFelxOS 1, a Debian-based distribution with Cinnamon, apt and xv4.sh (4.5 GB ISO, released 1 Feb 2026), its goals, screenshots and download.",
        "\"New: Install IFelxOS\" button on the home page.",
    ] },
    { v: "1.7", date: "2026-01-24", changes: [
        "New home page with the list of AXV projects: jowa-GPT, jowa-football-ai, jowa-e, jowa-ifelx, xv2-xv5, IFelxOS, IFelx-mini, IFelxUI and WEX browser.",
        "\"New: WEX browser\" link.",
    ] },
    { v: "1.6", date: "2026-01-15", changes: [
        "xv4t1 package channel with xv4.sh (5 Jan) and the axv, joda and xv3 packages (8 Jan).",
        "Home page: IFelxOS release date announced, author shown as AXV.",
    ] },
    { v: "1.5", date: "2026-01-04", changes: [
        "IFelx Shop grows to over 20 packages: apt_pkg_mgr 1.0 (14 Dec), 14 tools such as calculator_gui, notes_cli, password_safe, system_monitor and todo_cli (3 Jan), and bat, busybox, fd, jq and ncdu (4 Jan).",
    ] },
    { v: "1.4", date: "2025-12-04", changes: [
        "The site takes its name, IFelx Web (was IFlex Web).",
        "Bing Webmaster verification (3 Dec).",
    ] },
    { v: "1.3", date: "2025-11-28", changes: [
        "IFelx Shop: pages for the xv4 package manager (install, update, upgrade), an update list and the first packages (jotxt; cute.sh on 30 Nov).",
        "Site renamed from IFex Web to IFlex Web.",
        "Google Search verification.",
    ] },
    { v: "1.2", date: "2025-11-27", changes: [
        "arabicxv: an Arabic topic search page (26 Nov), linked from the home page as \"arabicxv Topic Search Engine\".",
        "Home page introduction and copyright line.",
    ] },
    { v: "1.1", date: "2025-11-23", changes: [
        "\"Official Site\" title, robots.txt and sitemap.xml for search engines (22 Nov).",
        "Site icon (favicon).",
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
versionBtn.setAttribute("aria-label", `IFelx Web version ${RELEASES[0].v}, open the release history`);
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
