import React, { Component } from 'react';
import PropTypes from 'prop-types';
import AppBar from '@material-ui/core/AppBar';
import Divider from '@material-ui/core/Divider';
import Drawer from '@material-ui/core/Drawer';
import IconButton from '@material-ui/core/IconButton';
import List from '@material-ui/core/List';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import MenuIcon from '@material-ui/icons/Menu';
import Main from './../Main/Main'
import TaskDrawer from './TaskDrawer';
import UserMenu from './UserMenu'
import classNames from 'classnames';
import { withStyles } from '@material-ui/core/styles';
import { styles } from './Styles/Header'

class Header extends Component {
  state = {
    open: false,
    selectedTask: 0,
    tasks: [],
  };

  handleDrawerOpen = () => {
    this.setState({ open: true });
  };

  handleDrawerClose = () => {
    this.setState({ open: false });
  };

  fetchData = () => {
    fetch('/api/backend/images')
    .then((data) => data.json())
    .then((tasks) => this.setState({tasks}, () => {
      console.log(tasks);
    }));
  }

  componentWillMount() {
    this.fetchData();
  }

  selectedTaskHandler = (selectedTask) => {
    this.setState({selectedTask});
  }

  render() {
    const { classes, theme } = this.props;

    return (
      <div className={classes.root}>
        <AppBar
          position="absolute"
          className={classNames(classes.appBar, this.state.open && classes.appBarShift)}
        >
          <Toolbar disableGutters={!this.state.open} className={classes.navBar}>
            <IconButton
              color="inherit"
              aria-label="Open drawer"
              onClick={this.handleDrawerOpen}
              className={classNames(classes.menuButton, this.state.open && classes.hide)}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="title" color="inherit" noWrap>
              Phase 3 Task Results
            </Typography>
            <UserMenu />
          </Toolbar>
        </AppBar>
        <Drawer
          variant="permanent"
          classes={{
            paper: classNames(classes.drawerPaper, !this.state.open && classes.drawerPaperClose),
          }}
          open={this.state.open}
        >
          <div className={classes.toolbar}>
            <IconButton onClick={this.handleDrawerClose}>
              {theme.direction === 'rtl' ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            </IconButton>
          </div>
          <Divider />
          <List><TaskDrawer tasks={this.state.tasks} selectedTaskHandler={this.selectedTaskHandler} /></List>
        </Drawer>
        <Main task={this.state.tasks[this.state.selectedTask]} />
      </div>
    );
  }
}

Header.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
};

export default withStyles(styles, { withTheme: true })(Header);
