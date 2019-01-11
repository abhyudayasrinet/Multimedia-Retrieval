import React, { Component } from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { withStyles } from '@material-ui/core/styles';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import Menu from '@material-ui/core/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import PersonIcon from '@material-ui/icons/Person';
import { styles } from './Styles/Header'

class UserMenu extends Component {
  state = {
    anchorEl: null,
  };

  componentWillMount() {
    this.setState({imageError: false});
  }

  handleClick = event => {
    this.setState({ anchorEl: event.currentTarget });
  };

  handleClose = (event) => {
    this.setState({ anchorEl: null });
  };

  render() {
    const { classes } = this.props;
    const { anchorEl } = this.state;

    return (
      <div>
        <Button
          aria-owns={anchorEl ? 'simple-menu' : null}
          aria-haspopup="true"
          onClick={this.handleClick}
        >
          <Avatar alt="User" className={classNames(classes.avatar, classes.bigAvatar)}>
            <PersonIcon />
          </Avatar>
        </Button>
        <Menu
          id="simple-menu"
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={this.handleClose}
        >
          <MenuItem onClick={this.handleClose}>Logout</MenuItem>
        </Menu>
      </div>
    );
  }
}

UserMenu.propTypes = {
  classes: PropTypes.object.isRequired,
  theme: PropTypes.object.isRequired,
};

export default withStyles(styles, { withTheme: true })(UserMenu);
