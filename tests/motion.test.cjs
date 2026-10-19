const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync('web/motion.js', 'utf8'), context);

test('vector arrows retain image displacement and direction across aspect ratios', () => {
  for (const [width, height] of [[640, 400], [400, 640], [1280, 720]]) {
    for (const vector of [[.01, 0], [-.01, 0], [0, .01], [0, -.01], [.01, .01], [-.01, -.01]]) {
      const g = context.vectorGeometry([.4, .4, .6, .6], vector, width, height, 4);
      assert.ok(g);
      assert.ok(Math.abs(g.end[0] - g.tail[0] - vector[0] * width * 4) < 1e-8);
      assert.ok(Math.abs(g.end[1] - g.tail[1] - vector[1] * height * 4) < 1e-8);
    }
  }
});
test('enlarged vectors stay inside the image and flag clipping', () => {
  const g = context.vectorGeometry([.01, .01, .03, .03], [.1, .1], 640, 400, 16);
  assert.ok(g.clipped);
  assert.ok(g.tail[0] >= 0 && g.tail[1] >= 0);
  assert.ok(g.tail[0] <= 640 && g.tail[1] <= 400);
});
test('arrows reject absent, invalid and sparse temporal evidence', () => {
  const d = {motion: {state: 'moving', vector: [.02, 0], gap_seconds: .1}};
  assert.ok(context.validVector(d));
  for (const motion of [undefined, {...d.motion, gap_seconds: 120}, {...d.motion, vector: [NaN, 0]}, {...d.motion, vector: [0, 0]}, {...d.motion, state: 'unknown'}]) {
    assert.ok(!context.validVector({motion}));
  }
  assert.equal(context.vectorGeometry([.4,.4,.5,.5], [0,0], 640,400,8), null);
});
test('description distinguishes image displacement from speed', () => {
  const text = context.motionDescription({motion: {state: 'moving', vector: [.02,0], direction: 'right', gap_seconds: .1, evidence: {displacement_px: 12.8}}});
  assert.match(text, /→ right on image/);
  assert.match(text, /12.8 analysis px \/ 0.10s/);
  assert.match(context.motionDescription({motion:{state:'unknown'}}), /no reliable vector/);
});
test('three-frame vectors require stable background evidence and bounded observation gaps',()=>{const d={motion:{state:'moving',vector:[.02,0],gap_seconds:10,evidence_kind:'three_frame_appearance',evidence:{frames:3,background_stable:true}}};assert.equal(context.validVector(d),true);d.motion.evidence.background_stable=false;assert.equal(context.validVector(d),false);d.motion.evidence.background_stable=true;d.motion.gap_seconds=90;assert.equal(context.validVector(d),false);});
