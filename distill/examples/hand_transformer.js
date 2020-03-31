function relu_single(x) {
    if (x < 0) {
        return 0;
    }
    return x;
}

function relu(x) {
    return math.map(x, relu_single);
}

function simple_sigmoid_single(x) {
    if (x < 0) {
        return 0;
    }
    if (x > 1) {
        return 1;
    }
    return x;
}

function simple_sigmoid(x) {
    return math.map(x, simple_sigmoid_single);
}

function softmax(x, options) {
    let axis = options.axis;
    if (axis != 0 && axis != 1) {
        console.log('bad axis');
        return null;
    }
    let ex = x.map(Math.exp);
    let sumex = math.apply(ex, axis, math.sum);
    let rv = ex.map(function(d, idx) {
        return d / (1e-12 + sumex.get([idx[1 - axis]]));
    });
    let sumrv = math.apply(rv, axis, math.sum);
    return rv;
}

function add_positional_embedding(vec_inputs, semes, clocks) {
    let rv = [];
    for (const i in vec_inputs) {
        let vec_input = vec_inputs[i];
        let len_input = math.size(vec_input).get([0]);
        for (let j=0; j < len_input; j++) {
            for (const k in clocks) {
                let clock = clocks[k];
                let cos = Math.cos(clock.speed * j);
                let sin = Math.sin(clock.speed * j);
                let kx = semes.indexOf(clock.semex);
                let ky = semes.indexOf(clock.semey);
                vec_input.set([j, kx], vec_input.get([j, kx]) + cos);
                vec_input.set([j, ky], vec_input.get([j, ky]) -+sin);
            }
        }
        rv.push(vec_input);
    }
    return rv;
}

function pos2mat(shift, semes, clocks) {
    let rv = str2mat('', semes);
    for (const i in clocks) {
        let clock = clocks[i];
        let cos = Math.cos(clock.speed * shift);
        let sin = Math.sin(clock.speed * shift);
        let jx = semes.indexOf(clock.semex);
        let jy = semes.indexOf(clock.semey);
        rv.set([jx, jx], rv.get([jx, jx]) + 10 * cos);
        rv.set([jx, jy], rv.get([jx, jy]) - 10 *sin);
        rv.set([jy, jx], rv.get([jy, jx]) + 10 * sin);
        rv.set([jy, jy], rv.get([jy, jy]) + 10 * cos);
    }
    return rv;
}

function str2clocks(s) {
    let clock_atoms = s.split(/[\s,]+/);
    let clocks = [];
    for (const i in clock_atoms) {
        let atom = clock_atoms[i];
        if (atom.match(/^[\s,]*$/)) {
            continue;
        }
        let coeff_str = atom.match(/[+\-0-9\.]*/)[0];
        let seme = atom.match(/[a-zA-Z]+/)[0];
        let clock = {speed:Number(coeff_str),
                     semex:seme + 'x',
                     semey: seme + 'y'};
        clocks.push(clock);
    }

    return clocks;
}

function tokens2vecs(tokens, semes) {
    let rows = [];
    for (const i in tokens) {
        let token = tokens[i];
        for (const s in token) {
            let vec = str2vec(token[s], semes);
            rows.push(vec);
        }
    }
    rows = math.matrix(rows);
    return rows;
}

function tokens2str(tokens, semes) {
    let strs = [];
    for (const i in tokens) {
        let token = tokens[i];
        for (const s in token) {
            let this_str = '[' + s + ': ' + token[s] + ']';
            strs.push(this_str);
        }
    }
    return strs.join('\n');
}

function tokens2words(tokens) {
    let rv = [];
    for (const i in tokens) {
        let token = tokens[i];
        for (const s in token) {
            rv.push(s)
        }
    }
    return rv
}

function str2vec(s, semes) {
    let atoms = s.split(/[\s,]+/);
    let rv = math.zeros(semes.length);
    for (const i in atoms) {
        let atom = atoms[i];

        if (atom.match(/^\s*$/)) {
            continue;
        }
        
        if (!atom.match(/^[+\-]?[0-9\.]*[a-zA-Z]+$/)) {
            console.log('bad atom', atom);
            continue;
        }

        let coeff_str = atom.match(/[+\-0-9\.]*/)[0];
        let seme = atom.match(/[a-zA-Z]+/)[0];
        let coeff = 1;
        if (coeff_str) {
            if (coeff_str == '+') {
                coeff = 1;
            } else if (coeff_str == '-') {
                coeff = -1;
            } else {
                coeff = Number(coeff_str);
            }
        }

        if (seme.length == 0) {
            continue;
        }
        let j = semes.indexOf(seme);
        if (j < 0) {
            throw 'error: unknown seme: ' + seme;
        }
        rv.set([j], rv.get([j]) + coeff);
    }
    return rv;
}

function str2mat(s, semes) {
    let atoms = s.split(/[\s,]+/);
    let rv = math.zeros(semes.length, semes.length);
    for (const k in atoms) {
        let atom = atoms[k];

        if (atom.match(/^\s*$/)) {
            continue;
        }

        if (!atom.match(/^[+\-]?[0-9\.]*[a-zA-Z]+>[a-zA-Z]+/)) {
            console.log('bad atom', atom);
            continue;
        }

        let coeff_str = atom.match(/[+\-0-9\.]*/)[0];
        let seme_i = atom.match(/[a-zA-Z]+>/)[0].slice(0, -1);
        let seme_j = atom.match(/>[a-zA-Z]+/)[0].slice(1);
        let coeff = 1;

        if (coeff_str) {
            if (coeff_str == '+') {
                coeff = 1;
            } else if (coeff_str == '-') {
                coeff = -1;
            } else {
                coeff = Number(coeff_str);
            }
        }
        
        let i = semes.indexOf(seme_i);
        let j = semes.indexOf(seme_j);

        if (i < 0) {
            console.log('bad seme', seme_i);
            continue;
        } 

        if (j < 0) {
            console.log('bad seme', seme_j);
            continue;
        } 
        
        rv.set([i, j], rv.get([i, j]) + coeff);
    }
    return rv;
}

function vec2str(v, semes) {
    let rv = '';
    math.forEach(v, function(val, idx) {
        if (val != 0) {
            let atom = Math.round(val* 1000) / 1000 + semes[idx] + ' ';
            if (val == 1) {
                atom = semes[idx] + ' ';
            }
            if (val == -1) {
                atom = '-' + semes[idx] + ' ';
            }
            if (val > 0) {
                atom = '+' + atom;
            }
            rv += atom; 
        }
    });
    return rv;
}

function mat2str(v, semes) {
    return 'a>b';
}

class HandLayer {
    constructor(args) {
        this.wrangler = args.wrangler;
        this.id = args.id;
        this.topLevel = args.topLevel;
        this.container = args.container;
        this.document = args.document;
        this.program = args.program
        
        if (this.topLevel) {
            this.textArea = this.document.createElement("textarea");
            this.container.appendChild(this.textArea);
            this.textArea.value = this.program;
            this.codeMirror = CodeMirror.fromTextArea(
                this.textArea, {
                    lineNumbers: true,
                    keyMap: "emacs",
                    mode: "yaml"
                });
            this.codeMirror.on("change", this.recompute.bind(this));
            this.codeMirror.setSize("100%", "500px");

            this.table = this.document.createElement('table')
            this.table.id = this.id + '-table';
            let header = this.document.createElement('thead');
            header.innerHTML = '<tr><th>Input</th><th>Output</th></tr>'
            this.container.appendChild(this.table);
            this.table.appendChild(header);
            this.table.appendChild(this.document.createElement('tbody'));
        }
    }

    redraw() {
        let selector = '#' + this.id + '-svg';

        d3.select(selector)
            .selectAll('line')
            .data(this.wires)
            .join('line')
            .attr("x1", d => d.x1)
            .attr("y1", d => d.y1)
            .attr("x2", d => d.x2)
            .attr("y2", d => d.y2)
            .attr('stroke', '#000');

        d3.select(selector)
            .selectAll('rect')
            .data(this.blocks)
            .join("rect")
            .attr("x", d => d.x)
            .attr("y", d => d.y)
            .attr("width", d => d.w)
            .attr("height", d => d.h)
            .attr("fill", "#ffb400")
        
        d3.select(selector)
            .selectAll('text')
            .data(this.blocks)
            .join('text')
            .text(d => d.text)
            .attr("x", d => d.x)
            .attr("y", d => d.y + (d.h / 2))
            .attr('fill', '#000');

        d3.select("#" + this.id + '-table')
            .select('tbody')
            .selectAll('tr')
            .data(this.get_io_pairs())
            .join('tr')
            .selectAll('td')
            .data(d => d)
            .join('td')
            .text(d => d);
    }
}

class HandDense extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:0, w: 50, h: 50, text: 'input'},
                       {x: 150, y:0, w:50, h:50, text: 'dense'},
                       {x: 300, y:0, w:50, h:50, text: 'output'},
                      ];
        this.wires = [
            {x1: 0, y1: 25, x2:150, y2: 25},
            {x1: 300, y1: 25, x2:150, y2: 25}
        ];
        this.recompute();
    }

    get_io_pairs() {
        let bias = this.bias;
        let mat = this.mat;
        let vec_inputs = this.inputs.map(x => str2vec(x, this.semes));
        let vec_outputs = vec_inputs.map(
            function (vec_input) {
                return math.add(
                    bias,
                    math.multiply(math.transpose(mat), vec_input));
            });
        let str_outputs = vec_outputs.map(x => vec2str(x, this.semes));
        let io_pairs = this.inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }
    
    recompute(event) {
        this.program = this.codeMirror.getValue();
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch {
            console.log('parse failure');
        }
        if (!failed) {
            this.mat = str2mat(doc.mat, this.semes);
            this.bias = str2vec(doc.bias, this.semes);
            this.inputs = doc.examples;
        }
        this.redraw();
    }
}


class HandFeedForward extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:0, w:50, h:50, text: 'input'},
                       {x: 100, y:0, w:50, h:50, text: 'dense'},
                       {x: 200, y:0, w:50, h:50, text: 'relu'},
                       {x: 300, y:0, w:50, h:50, text: 'dense'},
                       {x: 400, y:0, w:50, h:50, text: 'output'},
                      ];
        this.wires = [
            {x1: 0, y1: 15, x2:100, y2: 15},
            {x1: 100, y1: 15, x2:200, y2: 15},
            {x1: 200, y1: 15, x2:300, y2: 15},
            {x1: 300, y1: 15, x2:400, y2: 15},
        ];
        if (args.topLevel) {
            this.recompute();
        }
    }

    get_io_pairs() {
        let bias1 = this.bias1;
        let mat1 = this.mat1;
        let bias2 = this.bias2;
        let mat2 = this.mat2;
        let vec_inputs = this.inputs.map(x => str2vec(x, this.semes));
        let vec_outputs = vec_inputs.map(
            function (vec_input) {
                return math.add(
                    bias2,
                    math.multiply(
                        math.transpose(mat2),
                        relu(
                            math.add(
                                bias1,
                                math.multiply(
                                    math.transpose(mat1),
                                    vec_input)))));
            });
        let str_outputs = vec_outputs.map(x => vec2str(x, this.semes));
        let io_pairs = this.inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }

    do_computation(vec_inputs) {
        let bias1 = this.programObj.bias1;
        let mat1 = this.programObj.mat1;
        let bias2 = this.programObj.bias2;
        let mat2 = this.programObj.mat2;
        let vec_outputs = vec_inputs.toArray().map(
            function (vec_input) {
                return math.add(
                    bias2,
                    math.multiply(
                        math.transpose(mat2),
                        relu(
                            math.add(
                                bias1,
                                math.multiply(
                                    math.transpose(mat1),
                                    vec_input)))));
            });
        return vec_outputs;
    }
    
    recompute(event) {
        this.program = this.codeMirror.getValue();
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch {
            console.log('parse failure');
        }
        if (!failed) {
            this.mat1 = str2mat(doc.mat1, this.semes);
            this.bias1 = str2vec(doc.bias1, this.semes);
            this.mat2 = str2mat(doc.mat2, this.semes);
            this.bias2 = str2vec(doc.bias2, this.semes);
            this.inputs = doc.examples;
        }
        this.redraw();
    }

    setProgramObj(obj, semes) {
        this.programObj = {
            mat1: str2mat(obj.mat1, semes),
            bias1: str2vec(obj.bias1, semes),
            mat2: str2mat(obj.mat2, semes),
            bias2: str2vec(obj.bias2, semes),
        };
    }
}

class HandSelfAttention extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:70, w:50, h:50, text: 'input'},
                       {x: 100, y:0, w:50, h:50, text: 'key'},
                       {x: 100, y:70, w:50, h:50, text: 'query'},
                       {x: 100, y:140, w:50, h:50, text: 'value'},
                       {x: 200, y:70, w:50, h:50, text: 'attention'},
                       {x: 300, y:70, w:50, h:50, text: 'interpretant'},
                       {x: 400, y:70, w:50, h:50, text: 'output'},
                      ];

        this.wires = [
            {x1: 0, y1: 95, x2:125, y2: 25},
            {x1: 0, y1: 95, x2:125, y2: 95},
            {x1: 0, y1: 95, x2:125, y2: 165},
            {x1: 125, y1: 25, x2:225, y2: 95},
            {x1: 125, y1: 95, x2:225, y2: 95},
            {x1: 225, y1: 95, x2:325, y2: 95},
            {x1: 125, y1: 165, x2:325, y2: 95},
            {x1: 325, y1: 95, x2:425, y2: 95},
        ];


        this.do_computation = this.do_computation.bind(this);
        
        if (args.topLevel) {
            this.recompute();
        }
    }

    get_io_pairs() {
        let keymat = this.keymat;
        let querymat = this.querymat;
        let valuemat = this.valuemat;
        let semes = this.semes;
        let vec_inputs = this.inputs.map(x => tokens2vecs(x, this.semes));
        let str_inputs = this.inputs.map(x => tokens2str(x, this.semes));
        let words = this.inputs.map(x => tokens2words(x));

        vec_inputs = add_positional_embedding(vec_inputs,
                                              this.semes,
                                              this.clocks);
        
        let keys = vec_inputs.map(x => math.multiply(x, keymat));
        let queries = vec_inputs.map(x => math.multiply(x, querymat));
        let values = vec_inputs.map(x => math.multiply(x, valuemat));
        let attn = queries.map(
            (q, i) => math.multiply(q, math.transpose(keys[i])));
        attn = attn.map(m => softmax(m, {axis: 1}));
        let vec_outputs = attn.map((m, i) => math.multiply(m, values[i]));

        let str_outputs = vec_outputs.map(
            function (x, i) {
                let strs = math.map(x, function (y, idx) {
                    y = (Math.round(1000 * y) / 1000);
                    if (y != 0) {
                        if (y == 1) {
                            return '+' + semes[idx[1]];
                        }
                        if (y == -1) {
                            return '-' + semes[idx[1]];
                        }
                        return y + semes[idx[1]];
                    }
                    return '';
                });
                let arr = math.apply(strs, 1,
                                  function (a) {
                                      return a.join(' ').trim();
                                  });
                let arr2 = arr.map(function(a, j) {
                    if (a == '') {
                        return '';
                    }
                    return '[' + words[i][j] + ':' + a + ']';
                });
                return arr2.toArray().join(' ').trim();
            }
        );
        let io_pairs = str_inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }

    do_computation(vec_inputs) {
        let querymat = this.programObj.querymat;
        let keymat = this.programObj.keymat;
        let valuemat = this.programObj.valuemat;
        let queries = math.multiply(vec_inputs, querymat);
        let keys = math.multiply(vec_inputs, keymat);
        let values = math.multiply(vec_inputs, valuemat);
        let attn = math.multiply(queries, math.transpose(keys));
        attn = softmax(attn, {axis: 1});
        let interpretant = math.multiply(attn, values);
        return interpretant;
    }

    recompute(event) {
        this.program = this.codeMirror ? this.codeMirror.getValue() : null;
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch {
            console.log('parse failure');
        }
        if (!failed) {
            this.clocks = str2clocks(doc.clocks ? doc.clocks : '');
            this.keymat = str2mat(doc.keymat, this.semes);
            this.querymat = str2mat(doc.querymat, this.semes);
            this.valuemat = str2mat(doc.valuemat, this.semes);
            this.querypos = (doc.querypos !== undefined ?
                             pos2mat(doc.querypos, this.semes, this.clocks) :
                             null);
            this.keypos = (doc.keypos !== undefined ?
                           pos2mat(doc.keypos, this.semes, this.clocks) :
                           null);
            if (this.querypos) {
                this.querymat = math.add(this.querymat, this.querypos);
            }
            if (this.keypos) {
                this.keymat = math.add(this.keymat, this.keypos);
            }
            this.inputs = doc.examples;
        }
        if (this.topLevel) {
            this.redraw();
        }
    }

    setProgramObj(obj, semes) {
        this.programObj = {
            docstring: obj.docstring,
            querypos: obj.querypos || null,
            keypos: obj.keypos || null,
            querymat: str2mat(obj.querymat, semes),
            keymat: str2mat(obj.keymat, semes),
            valuemat: str2mat(obj.valuemat, semes),
        }
    }
}


class HandMultiheadSelfAttention extends HandLayer {
    do_computation(vec_inputs) {
        let querymat = this.querymat;
        let keymat = this.keymat;
        let valuemat = this.valuemat;
        let return_values = vec_inputs.map(x => x);
        for (const i in this.programObj.heads) {
            let interpretants = this.programObj.heads[i].do_computation(
                vec_inputs);
            return_values = math.add(
                return_values,
                interpretants
            );
        }
        return return_values;
    }

    setProgramObj(obj, semes) {
        this.programObj = {
            heads: obj.heads.map(
                function (head) {
                    head.type = 'HandSelfAttention'; 
                    return createLayer(head, semes);
                }
            )
        }
    }
}

class HandEncdecAttention extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:70, w:50, h:50, text: 'input'},
                       {x: 100, y:0, w:50, h:50, text: 'key'},
                       {x: 100, y:70, w:50, h:50, text: 'query'},
                       {x: 100, y:140, w:50, h:50, text: 'value'},
                       {x: 200, y:70, w:50, h:50, text: 'attention'},
                       {x: 300, y:70, w:50, h:50, text: 'interpretant'},
                       {x: 400, y:70, w:50, h:50, text: 'output'},
                      ];

        this.wires = [
            {x1: 0, y1: 95, x2:125, y2: 25},
            {x1: 0, y1: 95, x2:125, y2: 95},
            {x1: 0, y1: 95, x2:125, y2: 165},
            {x1: 125, y1: 25, x2:225, y2: 95},
            {x1: 125, y1: 95, x2:225, y2: 95},
            {x1: 225, y1: 95, x2:325, y2: 95},
            {x1: 125, y1: 165, x2:325, y2: 95},
            {x1: 325, y1: 95, x2:425, y2: 95},
        ];
        if (args.topLevel) {
            this.recompute();
        }
    }

    get_io_pairs() {
        let keymat = this.keymat;
        let querymat = this.querymat;
        let valuemat = this.valuemat;
        let semes = this.semes;
        let vec_inputs = this.inputs.map(x => tokens2vecs(x, this.semes));
        let str_inputs = this.inputs.map(x => tokens2str(x, this.semes));
        let words = this.inputs.map(x => tokens2words(x));
        let vec_outputs = vec_inputs.map(
            function (vec_input) {
                let keys = math.multiply(vec_input, keymat);
                let queries = math.multiply(vec_input, querymat);
                let values = math.multiply(vec_input, valuemat);
                let attn = math.multiply(queries, math.transpose(keys));
                attn = softmax(attn, {axis: 1})
                let interpretant = math.multiply(attn, values);
                return interpretant;
            }
        );
        let str_outputs = vec_outputs.map(
            function (x, i) {
                let strs = math.map(x, function (y, idx) {
                    if (y != 0) {
                        if (y == 1) {
                            return '+' + semes[idx[1]];
                        }
                        if (y == -1) {
                            return '-' + semes[idx[1]];
                        }
                        return y + semes[idx[1]];
                    }
                    return '';
                });
                let arr = math.apply(strs, 1,
                                  function (a) {
                                      return a.join(' ').trim();
                                  });
                let arr2 = arr.map(function(a, j) {
                    if (a == '') {
                        return '';
                    }
                    return '[' + words[i][j] + ':' + a + ']';
                });
                return arr2.toArray().join(' ').trim();
            }
        );
        let io_pairs = str_inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }
    
    recompute(event) {
        this.program = this.codeMirror.getValue();
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch {
            console.log('parse failure');
        }
        if (!failed) {
            this.keymat = str2mat(doc.keymat, this.semes);
            this.querymat = str2mat(doc.querymat, this.semes);
            this.valuemat = str2mat(doc.valuemat, this.semes);
            this.inputs = doc.examples;
        }
        this.redraw();
    }

    do_computation(vec_inputs, encoder_output) {
        let querymat = this.programObj.querymat;
        let keymat = this.programObj.keymat;
        let valuemat = this.programObj.valuemat;
        let keys = math.multiply(encoder_output, keymat);
        let queries = math.multiply(vec_inputs, querymat);
        let values = math.multiply(encoder_output, valuemat);
        let attn = math.multiply(queries, math.transpose(keys));
        attn = softmax(attn, {axis: 1})
        let interpretant = math.multiply(attn, values);
        return interpretant;
    }
    
    setProgramObj(obj, semes) {
        this.programObj = {
            docstring: obj.docstring,
            querypos: obj.querypos || null,
            keypos: obj.keypos || null,
            querymat: str2mat(obj.querymat, semes),
            keymat: str2mat(obj.keymat, semes),
            valuemat: str2mat(obj.valuemat, semes),
        }
    }
}

class HandMultiheadEncdecAttention extends HandLayer {
    do_computation(vec_inputs, encoder_output) {
        let return_values = vec_inputs.map(x => x);
        for (const i in this.programObj.heads) {
            let interpretants = this.programObj.heads[i].do_computation(
                vec_inputs, encoder_output);
            return_values = math.add(
                return_values,
                interpretants
            );
        }
        return return_values;
    }

    setProgramObj(obj, semes) {
        this.programObj = {
            heads: obj.heads.map(
                function (head) {
                    head.type = 'HandEncdecAttention'; 
                    return createLayer(head, semes);
                }
            )
        }
    }
}

class HandRnn extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:70, w:50, h:50, text: 'input'},
                       {x: 70, y:70, w:50, h:50, text: 'embed'},
                       {x: 140, y:70, w:50, h:50, text: 'rnn1'},
                       {x: 210, y:70, w:50, h:50, text: 'rnn2'},
                       {x: 280, y:70, w:50, h:50, text: 'pool'},
                       {x: 350, y:70, w:50, h:50, text: 'dense'},
                       {x: 420, y:70, w:50, h:50, text: 'output'},
                      ];

        this.wires = [
            {x1: 0, y1: 95, x2:425, y2: 95},
        ];
        this.recompute();
    }

    get_io_pairs() {
        let a1 = this.a1;
        let b1 = this.b1;
        let bias1 = this.bias1;
        let a2 = this.a2;
        let b2 = this.b2;
        let bias2 = this.bias2;
        let semes = this.semes;
        let lexicon = this.lexicon;
        let inputs = this.inputs.map(x => x.split(/\s+/));
        let vec_inputs = inputs.map(x => x.map(y => str2vec(lexicon[y], semes)));
        let vec_outputs = [];
        for (const i in vec_inputs) {
            let h0t = math.zeros(semes.length);
            let h1t = math.zeros(semes.length);
            let h2t = math.zeros(semes.length);
            let h2t_tot = math.zeros(semes.length);
            for (const t in vec_inputs[i]) {
                h0t = vec_inputs[i][t];
                h1t = simple_sigmoid(
                    math.add(
                        h0t,
                        math.add(
                            math.multiply(h0t, this.a1),
                            math.add(
                                math.multiply(h1t, this.b1),
                                this.bias1))));
                h2t = simple_sigmoid(
                    math.add(
                        h1t,
                        math.add(
                            math.multiply(h1t, this.a2),
                            math.add(
                                math.multiply(h2t, this.b2),
                                this.bias2))));
                h2t_tot = math.add(h2t_tot, h2t);
            }
            let x0 = math.multiply(h2t_tot, 1 / vec_inputs[i].length);
            let x1 = simple_sigmoid(
                math.add(math.multiply(x0, this.c), this.cbias));
            vec_outputs.push(x1);
        }
        let str_outputs = vec_outputs.map(vec => vec2str(vec, semes));
        let io_pairs = this.inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }
    
    recompute(event) {
        this.program = this.codeMirror.getValue();
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch(err) {
            console.log('parse failure');
        }
        if (!failed) {
            this.lexicon = doc.lexicon;
            this.a1 = str2mat(doc.rnn_layer1.A, this.semes);
            this.b1 = str2mat(doc.rnn_layer1.B, this.semes);
            this.bias1 = str2vec(doc.rnn_layer1.bias, this.semes);
            this.a2 = str2mat(doc.rnn_layer2.A, this.semes);
            this.b2 = str2mat(doc.rnn_layer2.B, this.semes);
            this.bias2 = str2vec(doc.rnn_layer2.bias, this.semes);
            this.c = str2mat(doc.dense1.C, this.semes);
            this.cbias = str2vec(doc.dense1.c, this.semes);
            this.inputs = doc.examples;
        }
        this.redraw();
    }
}

class HandTransformer extends HandLayer {
    constructor(args) {
        super(args);
        this.blocks = [{x: 0, y:70, w:50, h:50, text: 'input'},
                       {x: 420, y:70, w:50, h:50, text: 'output'},
                      ];

        this.wires = [
            {x1: 0, y1: 95, x2:425, y2: 95},
        ];
        this.recompute();
    }

    get_io_pairs() {
        let semes = this.semes;
        let encoder_lexicon = this.encoder_lexicon;
        let decoder_lexicon = this.decoder_lexicon;
        let inputs = this.inputs.map(x => x.split(/\s+/));
        let vec_inputs = inputs.map(x => x.map(y => str2vec(encoder_lexicon[y],
                                                            semes)));
        let str_outputs = [];
        for (const exid in inputs) {
            let example = inputs[exid];
            let encoder_outputs = vec_inputs[exid];
            for (const i in this.encoder_stack) {
                let layer = this.encoder_stack[i];
                encoder_outputs = layer.do_computation(encoder_outputs);
            }
            let decoder_inputs = ['sos'];
            let output = [];
            for (let t=0; t<20; t++) {
                let decoder_outputs = decoder_inputs.map(
                    y => str2vec(decoder_lexicon[y], semes));
                for (const i in this.decoder_stack) {
                    let layer = this.decoder_stack[i];
                    decoder_outputs = math.add(decoder_outputs,
                        layer.do_computation(decoder_outputs,
                                             encoder_outputs));
                    console.log(layer, decoder_outputs);
                }
                let maxLogit = -1e12;
                let chosen = null;
                for (const word in decoder_lexicon) {
                    let logit = math.dot(str2vec(decoder_lexicon[word],
                                                 semes),
                                         math.flatten(
                                             math.row(decoder_outputs, t))
                                        );
                    console.log(word, logit);
                    if (logit > maxLogit) {
                        maxLogit = logit;
                        chosen = word;
                    }
                }
                decoder_inputs.push(chosen);
                if (chosen == 'eos') {
                    break;
                }
            }
            decoder_inputs = decoder_inputs.splice(1);
            if (decoder_inputs.indexOf('eos') >= 0) {
                decoder_inputs = decoder_inputs.splice(
                    0,
                    decoder_inputs.indexOf('eos')
                );
            }
            str_outputs.push(decoder_inputs.join(' '));
        }
        
        let io_pairs = this.inputs.map(function(a, i) {
            return [a, str_outputs[i]];
        });
        return io_pairs;
    }
    
    recompute(event) {
        this.program = this.codeMirror.getValue();
        let failed = true;
        let doc = null;
        try {
            doc = jsyaml.load(this.program);
            this.semes = doc.semes.split(/\s+/);
            failed = false;
        } catch(err) {
            console.log('parse failure', err.message);
        }
        if (!failed) {
            this.architecture = doc.architecture;
            this.encoder_stack = [];
            this.decoder_stack = [];
            for (const i in this.architecture.encoder) {
                this.encoder_stack.push(
                    createLayer(doc[this.architecture.encoder[i]], this.semes)
                );
            }
            for (const i in this.architecture.decoder) {
                this.decoder_stack.push(
                    createLayer(doc[this.architecture.decoder[i]], this.semes)
                );
            }
            this.encoder_lexicon = doc.encoder_lexicon;
            this.decoder_lexicon = doc.decoder_lexicon;            
            this.inputs = doc.examples;
        }
        this.redraw();
    }
}


function createLayer(obj, semes) {
    obj.topLevel = false;
    let cls = eval(obj.type);
    let rv = new cls({topLevel: false});
    rv.setProgramObj(obj, semes)
    return rv;
}


function positionalAnimation(div) {
    let data = ['She', 'will', 'eat', 'a', 'very', 'small', 'red', 'apple'];
    let clocks = [
        {speed:0.1,
         color: 'red'},
        {speed:0.15,
         color: 'orange'},
        {speed:0.2,
         color: 'brown'},
        {speed:0.25,
         color: 'green'},
        {speed:0.3,
         color: 'blue'},
        {speed:0.35,
         color: 'purple'},
        {speed:0.4,
         color: 'black'},
    ];

    function selectWord(idx) {
        d3.select('#figure-positional-svg')
            .selectAll('text.words')
            .data(data)
            .classed('words', 'true')
            .join('text')
            .attr('x', (d, i) => 50 * (i + 1))
            .attr('y', 100)
            .attr('fill', function(d, i) {
                if (idx == i) {
                    return '#f00';
                }
                return '#000';
            })
            .text(d => d)
            .on('click', function(d, i) {
                selectWord(i);
            })

        d3.select('#figure-positional-svg')
            .selectAll('text.numbers')
            .data(data)
            .classed('numbers', 'true')
            .join('text')
            .attr('x', (d, i) => 50 * (i + 1))
            .attr('y', 130)
            .attr('fill', function(d, i) {
                if (idx == i) {
                    return '#f00';
                }
                return '#000';
            })
            .text((d, i) => i)
            .on('click', function(d, i) {
                selectWord(i);
            });
        
        d3.select('#figure-positional-svg')
            .selectAll('circle.clocks')
            .data(clocks)
            .classed('clocks', 'true')
            .join('circle')
            .attr('cx', (d, i) => 60 * (i + 1))
            .attr('cy', 50)
            .attr('r', 20)
            .attr('stroke', (d, i) => d.color)
            .attr('fill', '#fff');

        d3.select('#figure-positional-svg')
            .selectAll('line.clocks')
            .data(clocks)
            .classed('clocks', 'true')
            .join('line')
            .attr('x1', (d, i) => 60 * (i + 1))
            .attr('y1', 50)
            .attr('x2', (d, i) => 60 * (i + 1) + 20 * math.cos(d.speed * idx))
            .attr('y2', (d, i) => 50 + 20 * math.sin(d.speed * idx))
            .attr('stroke', (d, i) => d.color)
    }

    selectWord(0);
    
}
