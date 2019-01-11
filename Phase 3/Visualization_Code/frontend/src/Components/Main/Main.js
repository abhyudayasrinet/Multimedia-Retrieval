import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Clusters from './Clusters';
import { withStyles } from '@material-ui/core/styles';
import { styles } from '../Header/Styles/Header'

class Main extends Component {
  state = {
    task: {},
  }

  componentWillReceiveProps(nextProps) {
    this.setState({task: nextProps.task});
  }

  render() {
    const { classes } = this.props;

    return (
      <main className={classes.content}>
        <div className={classes.toolbar} />
        <Clusters task={this.state.task} />
      </main>
    );
  }
}

Main.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles, { withTheme: true })(Main);
