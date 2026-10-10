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
        "Official release history of IFelx Web, starting from 2.0.",
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
        "New design for the whole site in the IFelx identity: dark theme, blue accent, rounded cards and pill buttons. Replaces the original design.",
        "Shared navbar with a project quick search (press /) and a phone menu, and a shared footer on every page.",
        "Copy buttons on code blocks, a screenshot viewer, and cards that fade in while scrolling.",
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
                </div>
                <button type="button" class="ifx-changelog-close" aria-label="Close">${ICON.close || "×"}</button>
            </div>
            <ol class="ifx-changelog-list"></ol>
        </div>`;
    const list = dlg.querySelector(".ifx-changelog-list");
    RELEASES.forEach((r, i) => {
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
        list.appendChild(li);
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
