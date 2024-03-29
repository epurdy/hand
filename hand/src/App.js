import React from 'react';
import './App.css';
import axios from 'axios';

const INITIAL_SENTENCE = 'she will eat a very small red apple';

const colors = [
    "red", "green", "blue", "purple", "orange",
    "cyan", "magenta", "black",
    "burlywood",
    "rosybrown",
    "sandybrown",
    "goldenrod",
    "peru",
    "chocolate",
    "saddlebrown",
    "sienna",
    "brown",
    "maroon",
];

const levelHeight = 400;
const wordWidth = 200;

function Token(props) {
    let embeds = [];
    for (const index in props.embedding) {
        let embed = props.embedding[index];
        let y = props.y + 30 + index * 20;
        let valence = "positive";
        if (embed[0] === '-') {
            valence = "negative";
        } 
        embeds.push(
            <text filter="url(#solid)"
                  className={"Token-embedding " + valence} x={props.x} y={y}
                  key={index}
                  >{embed}</text>
        );
    }

    return (
        <g>
          <text filter='url(#solid)' x={props.x} y={props.y}>
            {props.surfaceForm}
          </text>
          {embeds}            
        </g>
    );
}

function Connection(props) {
    return (
        <g>
          <line {...props} className="Connection" />
          <circle cx={props.x1} cy={props.y1} r={4}
                  fill={props.stroke} />
          <circle cx={props.x2} cy={props.y2} r={4}
                  fill={props.stroke} />
        </g>
    );
}

function Layer(props) {
    if (!props.layer.tokens) {
        return (<text x="200" y="100">No sentence!</text>);
    }

    let tokens = [];
    let connections = [];
    let head_names = [];
    
    for (const index in props.layer.tokens) {
        let token = props.layer.tokens[index];
        let x = 200 + index * wordWidth;
        let y = 100 + props.level * levelHeight;
        tokens.push(<Token surfaceForm={token} x={x} y={y} key={index}
                    embedding={props.layer.embeddings[index]}
                    />);
    }

    for (const hidx in props.layer.heads) {
        if (props.layer.heads[hidx]) {
            head_names.push(
                <text stroke={colors[hidx]}
                      key={"left-" + props.layer.heads[hidx]}
                      className="Layer-headname"
                      onMouseEnter={() => props.changeInfo(
                          props.layer.heads[hidx] + "\n\n" +
                  props.layer.head_descs[hidx])}
                  onMouseLeave={() => props.changeInfo("")}
                  x={30}
                  y={-180 + (props.level * levelHeight) + ((hidx + 1) * 3)}>
                  {props.layer.heads[hidx]}
                </text>
            );

            head_names.push(
                <text stroke={colors[hidx]}
                      key={"right-" + props.layer.heads[hidx]}
                      className="Layer-headname"
                      onMouseEnter={() => props.changeInfo(
                          props.layer.heads[hidx] + "\n\n" +
                  props.layer.head_descs[hidx])}
                  onMouseLeave={() => props.changeInfo("")}
                  x={(tokens.length + 1) * wordWidth}
                  y={-180 + (props.level * levelHeight) + ((hidx + 1) * 3)}>
                  {props.layer.heads[hidx]}
                </text>
            );
        }
    }

    for (const index in props.layer.connections) {
        let connection = props.layer.connections[index];
        for (const index2 in connection) {
            let prev = connection['in'];
            let cur = connection['out'];
            let x1 = 230 + prev * wordWidth;
            let x2 = 230 + cur * wordWidth;
            let y1 = 100 + (props.level - 1) * levelHeight;
            let y2 = 100 + props.level * levelHeight;
            connections.push(
                <Connection x1={x1} y1={y1} x2={x2} y2={y2}
                            onMouseEnter={() => props.changeInfo(connection.info)}
                  onMouseLeave={() => props.changeInfo("")}
                  stroke={colors[connection.hidx]} 
                  key={connection.hidx + " " + index + " " + index2} />
            );
        }
    }
    
    return (
        <g className="Layer">
          <text className="Layer-name" x={20}
                y={-200 + (props.level * levelHeight)}
                onMouseEnter={() => props.changeInfo(props.layer.desc)}
            onMouseLeave={() => props.changeInfo("")}
            >
            {props.layer.name}
          </text>
          <text className="Layer-name" x={(tokens.length + 1) * wordWidth}
                y={-200 + (props.level * levelHeight)}>
            {props.layer.name}
          </text>
          {head_names}
          <g className="Connections">
            {connections}
          </g>
          <g className="Tokens">
            {tokens}
          </g>
        </g>
    );
}

class Program extends React.Component {
    render() {
        return (
            <div className="Program">
              <textarea value={this.props.program}
                        readOnly={true}
                        rows={60} cols={120}
                        style={{display: this.props.editing ? "block" : "none"}}
                        />
            </div>
        );
    }
}

class Input extends React.Component {
    constructor(props) {
        super(props);
        this.state = {value: props.initialValue};

        this.changeSentence = props.changeSentence;
        this.handleChange = this.handleChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleRandom = this.handleRandom.bind(this);
    }

    handleChange(event) {
        this.setState({value: event.target.value});
    }
    
    handleSubmit(event) {
        event.preventDefault();
        this.changeSentence(this.state.value);
    }

    handleRandom(event) {
        event.preventDefault();
        this.changeSentence("");
    }
    
    render() {
        return (
            <div>
              <form onSubmit={this.handleSubmit} className="SentenceEntry">
                <label>
                  Sentence:
                  <input type="text" className="SentenceEntry-text" value={this.state.value} onChange={this.handleChange} />
                </label>
                <input type="submit" value="Submit" />
              </form><br />
              <form onSubmit={this.handleRandom} className="SentenceEntry">
                <input type="submit" value="Random" />
              </form>
            </div>
        );
    }
}

function Header(props) {
    return (
        <div className="Header">
          <Input initialValue={INITIAL_SENTENCE}
                 changeSentence={props.changeSentence}
                 />
          <Program program={props.program}
                   editing={props.editing} />
          <h1>Legible Transformers</h1>
          <h3>by Eric Purdy</h3>
          <h3>Have you ever wondered what goes on inside Transformer?</h3>
          <button className="link-button"
                  onClick={props.showExplanation} >What is this?</button>
        </div>
    );
}

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {layers: [],
                      editing: false,
                      program: "",
                      infoboxContent: ""};

        this.changeInfo = this.changeInfo.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);
    }

    componentDidMount() {
        this.changeSentence(INITIAL_SENTENCE);
    }
    
    changeSentence(sentence) {
        let random = !sentence;
        
        axios.post('http://localhost:8000/parse',
                   {crossDomain: true,
                    sentence: sentence,
                    random: random})
            .then(response => {
                this.setState(
                    {...response.data}
                )})
            .catch(() => console.log('oh dear'));
    }

    changeInfo(info) {
        this.setState({infoboxContent: info});
    }

    showExplanation(info) {
        this.setState({showExplanation: !this.state.showExplanation});
    }
    
    handleKeyPress(e) {
        if (e.keyCode === 69) { // e
            this.setState({editing: !this.state.editing})
        }
        console.log(e.keyCode, this.state.editing);
    }
    
    render() {
        let layers = [];
        
        for (const index in this.state.layers) {
            let layer = this.state.layers[index];
            layers.push(
                <Layer layer={layer} key={index} level={index}
                       changeInfo={this.changeInfo} />
            );
        }

        layers = layers.reverse();

        let mainpart = null;
        
        if (this.state.error || this.state.layers[0] === undefined) {
            mainpart = (
                <h1 className="Error">{this.state.error}</h1>
            );
        } else if (this.state.showExplanation) {
            mainpart = (
                <div className="Explanation">
                  <p>Transformer is a neural network architecture that is extremely effective for NLP (natural language processing) tasks.
                    The best introduction to Transformer is probably &nbsp;
                    <a href="http://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer</a>.
                  </p>

                  <p>
                    This page shows our work on "legible" Transformers.
                    These are Transformers whose weights have been set by
                    hand in a human-comprehensible way. 
                  </p>
                </div>
            );
        } else {
            mainpart = (
                <svg className="Svg"
                     width={(2 + this.state.layers[0].tokens.length) * wordWidth}
                     height={(this.state.layers.length + 1) * levelHeight} >
                  <defs>
                    <filter x="0" y="0" width="1" height="1" id="solid">
                      <feFlood floodColor="white"/>
                      <feComposite in="SourceGraphic" operator="over" />
                    </filter>
                  </defs>
                  {layers}
                </svg>
            );
        }
        
        return (
            <div className="App">

              <a href="https://github.com/epurdy/hand"><img width="149" height="149" src="https://github.blog/wp-content/uploads/2008/12/forkme_left_darkblue_121621.png?resize=149%2C149" className="fork" alt="Fork me on GitHub" data-recalc-dims="1" />foo</a>
              
              <Header initialSentence={INITIAL_SENTENCE}
                      changeSentence={this.changeSentence.bind(this)}
                      showExplanation={this.showExplanation.bind(this)}
                      program={this.state.program}
                      editing={this.state.editing} />

              {mainpart}
              <div className="Infobox" id="Infobox">
                {this.state.infoboxContent}
              </div>
            </div>
        );
    }
}

export default App;
