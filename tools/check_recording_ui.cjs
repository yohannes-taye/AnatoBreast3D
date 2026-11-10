const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || path.resolve(__dirname, '../../tools/photogrammetry/preview-browser/node_modules/playwright'));

(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless:true, args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1200}});
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.goto(process.env.VIEWER_URL || 'http://127.0.0.1:8765/viewer/recording.html');
    await page.locator('#connection').filter({hasText:'Both cameras connected'}).waitFor({timeout:40000});
    await page.waitForFunction(()=>document.getElementById('cam1rgb').naturalWidth>0 && document.getElementById('cam2rgb').naturalWidth>0);
    if (!await page.locator('#record').isEnabled()) throw new Error('Recording is unavailable despite a working preview');
    await page.locator('#overlay').fill('35'); await page.locator('#overlay').dispatchEvent('input');
    if (await page.locator('#cam1depth').evaluate(e=>e.style.opacity) !== '0.35') throw new Error('Depth overlay did not update');
    await page.locator('#overlay').fill('0'); await page.locator('#overlay').dispatchEvent('input');
    await page.locator('#cam2visible').uncheck(); await page.locator('#cam2visible').check();
    await page.screenshot({path:'runs/point_tracking/recording_ui.png',fullPage:true});
    if (await page.locator('#showTracks').isVisible()) {
      await page.locator('#showTracks').click();
      await page.locator('#playback').waitFor();
      await page.locator('#frame').fill('5'); await page.locator('#frame').dispatchEvent('input');
      await page.locator('#frameText').filter({hasText:'Frame 6/'}).waitFor();
      await page.screenshot({path:'runs/point_tracking/tracking_ui.png',fullPage:true});
      await page.locator('#showOverlap').click();
      await page.locator('#sceneTitle').filter({hasText:'3D camera overlap'}).waitFor();
    }
    if (process.argv.includes('--record')) {
      await page.locator('#duration').fill('2');
      await page.locator('#record').click();
      await page.locator('#recordState').filter({hasText:'recorded'}).waitFor({timeout:30000});
      const state = await page.request.get(new URL('/api/point-tracking/status', page.url()).href);
      const payload = await state.json();
      if (!payload.preview.session || payload.preview.frames < 1) throw new Error('The UI recording did not save usable frames');
      console.log('UI recording saved: '+payload.preview.session);
      await page.locator('#preview').click();
      await page.locator('#connection').filter({hasText:'Both cameras connected'}).waitFor({timeout:40000});
    }
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(JSON.stringify({ok:true,camera1:await page.locator('#cam1meta').textContent(),camera2:await page.locator('#cam2meta').textContent(),drift:await page.locator('#drift').textContent(),browserErrors:errors}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
