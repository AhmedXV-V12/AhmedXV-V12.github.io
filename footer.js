(function () {
    const IFX  = window.IFX || { icon: {}, engineUrl: "https://ifelx.tailce0b52.ts.net" };
    const ICON = IFX.icon;
    const year = new Date().getFullYear();

    const footer = document.createElement('footer');
    footer.className = 'site-footer';
    footer.innerHTML = `
        <div class="site-footer-inner">

            <div class="site-footer-brand">
                <a href="/index.html" class="site-footer-brand-row">
                    <img src="/logo.png" alt="" width="40" height="40" loading="lazy">
                    <span class="site-footer-brand-name">IFelx Web</span>
                </a>
                <p class="site-footer-desc">
                    Official website of AXV — open-source AI tools, operating systems, and technical projects.
                </p>
                <div class="site-footer-social">
                    <a href="https://www.instagram.com/axv.ifelx" target="_blank" rel="noopener" aria-label="Instagram">${ICON.instagram || ''}</a>
                    <a href="https://github.com/ahmedxv-v12" target="_blank" rel="noopener" aria-label="GitHub">${ICON.github || ''}</a>
                </div>
            </div>

            <div class="site-footer-col">
                <h4>Projects</h4>
                <a href="/jowa-gpt/index.html">Jowa-mAi</a>
                <a href="/jowa-football-ai/index.html">Jowa Football AI</a>
                <a href="/ifelx/index.html">IFelxOS</a>
                <a href="${IFX.engineUrl}" target="_blank" rel="noopener">IFelx Engine</a>
            </div>

            <div class="site-footer-col">
                <h4>Resources</h4>
                <a href="/ifelx/index.html">Download IFelxOS</a>
                <a href="/jowa-football-ai/v2/index.html">Download Football AI v2</a>
                <a href="/jowa-gpt/index.html">Download Jowa-mAi</a>
                <a href="/IFelx-shop/xv4/index.html">xv4 Package Manager</a>
            </div>

            <div class="site-footer-col">
                <h4>Connect</h4>
                <a href="https://www.instagram.com/axv.ifelx" target="_blank" rel="noopener">Instagram</a>
                <a href="https://github.com/ahmedxv-v12" target="_blank" rel="noopener">GitHub</a>
            </div>

        </div>

        <div class="site-footer-bottom">
            <span class="site-footer-copy">&copy; 2025–${year} IFelx Web — AXV</span>
            <div class="site-footer-bottom-links">
                <a href="/index.html">Home</a>
                <a href="${IFX.engineUrl}" target="_blank" rel="noopener">IFelx Engine</a>
                <button type="button" class="site-footer-top">${ICON.up || ''}Back to top</button>
            </div>
        </div>
    `;
    document.body.appendChild(footer);

    footer.querySelector('.site-footer-top').addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
})();
