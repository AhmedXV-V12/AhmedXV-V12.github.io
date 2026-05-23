(function () {
    const footer = document.createElement('footer');
    footer.className = 'site-footer';
    footer.innerHTML = `
        <div class="site-footer-inner">

            <div class="site-footer-brand">
                <img src="./logo.png" alt="IFelx Logo">
                <span class="site-footer-brand-name">IFelx Web</span>
                <p class="site-footer-desc">
                    Official website of AXV (Ahmed Walid) — open-source AI tools, operating systems, and technical projects.
                </p>
            </div>

            <div class="site-footer-col">
                <h4>Projects</h4>
                <a href="/jowa-gpt/index.html">jowamAi</a>
                <a href="/jowa-football-ai/index.html">Jowa Football AI</a>
                <a href="/ifelx/index.html">IFelxOS</a>
                <a href="https://ifelx.tailce0b52.ts.net/wex/">WEX Engine</a>
            </div>

            <div class="site-footer-col">
                <h4>Resources</h4>
                <a href="/ifelx/index.html">Download IFelxOS</a>
                <a href="/jowa-football-ai/v2/index.html">Download Football AI v2</a>
                <a href="/jowa-gpt/index.html">Download jowamAi</a>
            </div>

            <div class="site-footer-col">
                <h4>Connect</h4>
                <a href="https://www.instagram.com/axv.ifelx" target="_blank">Instagram</a>
                <a href="https://github.com/ahmedxv-v12" target="_blank">GitHub</a>
            </div>

        </div>

        <div class="site-footer-bottom">
            <span class="site-footer-copy">&copy; 2025/2026 IFelx Web — AXV (Ahmed Walid)</span>
            <div class="site-footer-bottom-links">
                <a href="/index.html">Home</a>
                <a href="https://ifelx.tailce0b52.ts.net/wex/">WEX</a>
            </div>
        </div>
    `;
    document.body.appendChild(footer);
})();
