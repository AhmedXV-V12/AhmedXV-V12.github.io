(function () {
    const navbar = document.createElement('nav');
    navbar.className = 'navbar';
    navbar.innerHTML = `
        <a href="/index.html" class="navbar-brand">
            <img src="/logo2.png" alt="IFelx Logo">
            <span class="navbar-brand-name">IFelx Web</span>
        </a>
        <div class="navbar-search">
            <input type="text" id="navSearch" placeholder="Search projects..." oninput="
                const q = this.value.toLowerCase().trim();
                document.querySelectorAll('#projectsGrid .project-btn').forEach(btn => {
                    const name = btn.getAttribute('data-name') || btn.textContent.toLowerCase();
                    btn.style.display = (!q || name.includes(q)) ? '' : 'none';
                });
            " />
        </div>
        <div class="navbar-links">
            <a href="/jowa-gpt/index.html">Jowa-GPT</a>
            <a href="/jowa-football-ai/index.html">Football AI</a>
            <a href="/ifelx/index.html">IFelxOS</a>
            <a href="https://wex-br.github.io">WEX</a>
            <a href="https://www.instagram.com/axv.bin" target="_blank" class="navbar-instagram">
                <img src="https://static.cdninstagram.com/rsrc.php/v4/yI/r/VsNE-OHk_8a.png" alt="Instagram" class="navbar-insta-icon">
                axv.bin
            </a>
        </div>
    `;
    document.body.insertBefore(navbar, document.body.firstChild);
})();
