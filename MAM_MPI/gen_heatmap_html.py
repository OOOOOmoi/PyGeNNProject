#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 17 群体放电率热力图 HTML（横轴 gL，纵轴 vipext），自适应色标。"""
import json

d = json.load(open('vipgl_2d_grid_result.json'))
pops = d['pops']
gl = [float(x) for x in d['gl']]
vipext = [float(x) for x in d['vipext']]
rates = d['rates']

# 群体类型分组着色标题
type_map = {'H': 'H (thalamic)', 'E': 'E (exc)', 'P': 'P (PV)', 'S': 'S (SST)', 'V': 'V (VIP)'}

# 生成热力图格子
def heatmap_cell(pop, r):
    # 每群体自适应色标：0 -> 深蓝，max -> 红
    pass

# 用 JS 渲染，数据内嵌 JSON
data_js = json.dumps(d, ensure_ascii=False)

html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>17 群体放电率热力图 (gL × vipext)</title>
<style>
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin: 20px; background: #fff; color: #1a1a1a; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px 0; }
  .subtitle { font-size: 13px; color: #666; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
  .panel { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; }
  .panel-title { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  .panel-sub { font-size: 11px; color: #999; margin-bottom: 8px; }
  .hm { position: relative; }
  .hm canvas { display: block; width: 100%; border-radius: 4px; }
  .axis-label { font-size: 10px; color: #888; text-align: center; margin-top: 4px; }
  .legend { margin-top: 20px; padding: 12px; border: 1px solid #e0e0e0; border-radius: 8px; }
  .legend-title { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
  .legend-row { font-size: 11px; color: #666; line-height: 1.5; }
</style>
</head>
<body>
<h1>17 群体放电率热力图</h1>
<div class="subtitle">横轴 = VIP 漏电导缩放 gL (1.0~2.0)，纵轴 = VIP 外部输入缩放 vipext (0.05~1.0)。每群体独立<strong>对数色标</strong>（log10，0=深蓝，群体内最大=黄），突出低值区的相变分界线。</div>
<div class="grid" id="grid"></div>
<div class="legend">
  <div class="legend-title">说明</div>
  <div class="legend-row">
    H=丘脑输入 · E=兴奋性锥体 · P=PV 快放电中间神经元 · S=SST 中间神经元 · V=VIP 中间神经元。<br>
    每张热力图右侧渐变条为该群体放电率的<strong>对数色标</strong>（单位 Hz），顶部为该群体最大放电率。<br>
    颜色越黄 = 放电率越高，越深蓝 = 越接近 0。相变分界表现为一条沿 vipext 方向的亮暗突变横带（例如 V4/S4 在 vipext≈0.30~0.35 处的陡峭跳变）。
  </div>
</div>
<script>
const DATA = __DATA__;
const GL = DATA.gl.map(Number);
const VIP = DATA.vipext.map(Number);
const POPS = DATA.pops;
const RATES = DATA.rates;
const TYPE_LABEL = {H:'H 型 (丘脑)', E:'E 型 (兴奋)', P:'P 型 (PV)', S:'S 型 (SST)', V:'V 型 (VIP)'};

function renderPanel(pop){
  const M = RATES[pop];           // 21 x 20 (行=gL, 列=vipext)
  const t = pop[0];
  const panel = document.createElement('div');
  panel.className = 'panel';
  const title = document.createElement('div');
  title.className = 'panel-title';
  title.textContent = pop + '  (' + TYPE_LABEL[t] + ')';
  const sub = document.createElement('div');
  const maxR = Math.max(...M.flat());
  sub.className = 'panel-sub';
  sub.textContent = '峰值 ' + maxR.toFixed(1) + ' Hz';
  panel.appendChild(title);
  panel.appendChild(sub);

  const hm = document.createElement('div');
  hm.className = 'hm';
  const cv = document.createElement('canvas');
  const W = 300, H = 260, padL = 44, padR = 14, padT = 10, padB = 30;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const nRow = M.length, nCol = M[0].length;
  const cw = plotW / nCol, ch = plotH / nRow;

  // 颜色映射：对数色标 (log10)，突出低值区的相变跳变。viridis 色带 蓝->青->绿->黄。
  const LOG_MAX = Math.log10(maxR + 1);
  function logt(v){ return LOG_MAX > 0 ? Math.log10(v + 1) / LOG_MAX : 0; }
  function color(t){
    t = Math.max(0, Math.min(1, t));
    // viridis 5 段近似: 蓝(0) -> 青(0.25) -> 绿(0.5) -> 黄(0.75) -> 黄绿亮(1)
    const stops = [
      [68, 1, 84],    // 深蓝紫
      [59, 82, 139],  // 蓝
      [33, 145, 140], // 青
      [94, 201, 98],  // 绿
      [253, 231, 37]  // 黄
    ];
    const seg = t * (stops.length - 1);
    const i0 = Math.min(stops.length - 2, Math.floor(seg));
    const f = seg - i0;
    const c = stops[i0].map((v, k) => Math.round(v + (stops[i0+1][k] - v) * f));
    return 'rgb('+c[0]+','+c[1]+','+c[2]+')';
  }

  for(let i=0;i<nRow;i++){
    for(let j=0;j<nCol;j++){
      const v = M[i][j];
      const t = logt(v);
      ctx.fillStyle = color(t);
      ctx.fillRect(padL + j*cw, padT + i*ch, cw, ch);
    }
  }
  // 网格线
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 0.5;
  for(let i=0;i<=nRow;i++){ ctx.beginPath(); ctx.moveTo(padL, padT+i*ch); ctx.lineTo(padL+plotW, padT+i*ch); ctx.stroke(); }
  for(let j=0;j<=nCol;j++){ ctx.beginPath(); ctx.moveTo(padL+j*cw, padT); ctx.lineTo(padL+j*cw, padT+plotH); ctx.stroke(); }

  // 轴刻度
  ctx.fillStyle = '#666'; ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
  const yTicks = [0, 5, 10, 15, 19];
  yTicks.forEach(k => { ctx.fillText(VIP[k].toFixed(2), padL-4, padT + k*ch + ch/2 + 3); });
  ctx.textAlign = 'center';
  const xTicks = [0, 5, 10, 15, 20];
  xTicks.forEach(k => { ctx.fillText(GL[k].toFixed(1), padL + k*cw + cw/2, padT + plotH + 12); });
  ctx.textAlign = 'center';
  ctx.fillStyle = '#888';
  ctx.save(); ctx.translate(10, padT+plotH/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('vipext', 0, 0); ctx.restore();
  ctx.fillText('gL', padL+plotW/2, H-4);

  // 色标条
  const barX = W-10, barY = padT, barH = plotH;
  for(let i=0;i<barH;i++){
    ctx.fillStyle = color(i/barH);
    ctx.fillRect(barX, barY+i, 8, 1);
  }
  ctx.fillStyle = '#666'; ctx.font='8px sans-serif'; ctx.textAlign='left';
  ctx.fillText(maxR.toFixed(0), barX+10, barY+4);
  ctx.fillText('0', barX+10, barY+barH);

  hm.appendChild(cv);
  panel.appendChild(hm);
  return panel;
}

const grid = document.getElementById('grid');
POPS.forEach(p => grid.appendChild(renderPanel(p)));
</script>
</body>
</html>
"""

html = html.replace('__DATA__', data_js)
with open('heatmap_17pop_gl_vipext.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("已生成 heatmap_17pop_gl_vipext.html")
