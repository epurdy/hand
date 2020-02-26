function relu_single(x) {
    if (x < 0) {
        return 0;
    }
    return x;
}

function relu(x) {
    return math.map(x, relu_single);
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

function tokens2vecs(tokens, semes) {
    //let rv = math.zeros(tokens.length, semes.length);
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
            throw 'error: unknown seme';
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
            let atom = val + semes[idx] + ' ';
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
        this.recompute();
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
}

class HandAttention extends HandLayer {
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
        this.recompute();
    }

    get_io_pairs() {
        let keymat = this.keymat;
        let querymat = this.querymat;
        let valuemat = this.valuemat;
        let semes = this.semes;
        let inputs = this.inputs;
        let vec_inputs = this.inputs.map(x => tokens2vecs(x, this.semes));
        let str_inputs = this.inputs.map(x => tokens2str(x, this.semes));
        let words = this.inputs.map(x => tokens2words(x));
        let vec_outputs = vec_inputs.map(
            function (vec_input) {
                let keys = math.multiply(vec_input, keymat);
                let queries = math.multiply(vec_input, querymat);
                let values = math.multiply(vec_input, valuemat);
                let attn = math.multiply(queries, math.transpose(keys));
                let interpretant = math.multiply(attn, values);
                attn = softmax(attn, {axis: 1})
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
}

