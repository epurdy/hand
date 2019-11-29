import React from 'react';
import './App.css';
import axios from 'axios'

const colors = [
    "red", "green", "blue", "yellow", "purple", "orange",
];

const levelHeight = 250;
const wordWidth = 200;

function Token(props) {
    let embeds = [];
    for (const index in props.embedding) {
        let embed = props.embedding[index];
        let y = props.y + 30 + index * 20;
        embeds.push(
                <text filter="url(#solid)"
            className="Token-embedding" x={props.x} y={y}
            key={index}
                >{embed}</text>
        );
    }

    return (
            <g>
            <circle cx={props.x + 30} cy={props.y} r={4} fill="red" />
            <text filter='url(#solid)' x={props.x} y={props.y}>
            {props.surfaceForm}
        </text>
            {embeds}            
        </g>
    );
}

class Connection extends React.Component {
    constructor(props) {
        super(props);
        this.state = {descriptionHidden: true};
    }

    render() {
        let rectx = (this.props.x1 + this.props.x2) / 2;
        let recty = (this.props.y1 + this.props.y2) / 2;
        return (
            <g>
                <line {...this.props} className="Connection"
                />
                <rect style={{display: this.state.descriptionHidden ?
                              'none' : ''
                             }}
            x={rectx} y={recty}
            width={300} height={300}
                fill="green"/>
                </g>
        );
    }
        
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
        if (props.layer.heads[hidx].name) {
            head_names.push(
                    <text stroke={colors[hidx]}
                key={props.layer.heads[hidx].name}
                className="Layer-headname"
                x={30}
                y={-80 + (props.level * levelHeight) + ((hidx + 1) * 3)}>
                    {props.layer.heads[hidx].name}
                </text>
            );
        }

        for (const index in props.layer.heads[hidx].connections) {
            let connection = props.layer.heads[hidx].connections[index];
            for (const index2 in connection) {
                let prev = connection[index2];
                let x1 = 230 + prev * wordWidth;
                let x2 = 230 + index * wordWidth;
                let y1 = 100 + (props.level - 1) * levelHeight;
                let y2 = 100 + props.level * levelHeight;
                connections.push(
                        <Connection x1={x1} y1={y1} x2={x2} y2={y2}
                    onMouseEnter={() => props.changeInfo(x1 + " " + y1)}
                    onMouseLeave={() => props.changeInfo("")}
                    stroke={colors[hidx]} 
                    key={hidx + " " + index + " " + index2} />
                );
            }
        }
    }
        
    return (
            <g className="Layer">
            <text className="Layer-name" x={20}
        y={-100 + (props.level * levelHeight)}>
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

class Input extends React.Component {
    constructor(props) {
        super(props);
        this.state = {value: props.initialValue}

        this.changeSentence = props.changeSentence
        this.handleChange = this.handleChange.bind(this)
        this.handleSubmit = this.handleSubmit.bind(this)
    }

    handleChange(event) {
        this.setState({value: event.target.value});
    }

    handleSubmit(event) {
        event.preventDefault();
        this.changeSentence(this.state.value)
    }
    
    render() {
        return (
                <form onSubmit={this.handleSubmit} className="SentenceEntry">
                <label>
                Sentence:
                <input type="text" value={this.state.value} onChange={this.handleChange} />
                </label>
                <input type="submit" value="Submit" />
                </form>
        );
    }
}

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {layers: [],
                      infoboxContent: ""};

        this.changeInfo = this.changeInfo.bind(this);
    }

    componentDidMount() {
        this.changeSentence('he ate a red apple');
    }
        
    changeSentence(sentence) {
        axios.post('http://localhost:8000/parse',
                   {crossDomain: true,
                    sentence: sentence})
            .then(response => this.setState(
                {layers: response.data.layers}))
            .catch(() => console.log('oh dear'))
    }

    changeInfo(info) {
        this.setState({infoboxContent: info});
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
        
        return (
                <div className="App">
                <Input initialValue="he ate a red apple"            
            changeSentence={this.changeSentence.bind(this)}
                />
                <svg className="Svg">
                <defs>
                <filter x="0" y="0" width="1" height="1" id="solid">
                <feFlood floodColor="white"/>
                <feComposite in="SourceGraphic" operator="over" />
                </filter>
                </defs>
                {layers}
            </svg>
                <div className="Infobox" id="Infobox">
                {this.state.infoboxContent}
                </div>
                </div>
        );
    }
}

export default App;
